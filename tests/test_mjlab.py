"""Optional native mjlab integration tests; run in integrations/mjlab environment."""

import numpy as np
import pytest

pytest.importorskip("mjlab")
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg

from mujoco_viser_usage.mjlab import mdp
from mujoco_viser_usage.mjlab.assets import export_scene
from mujoco_viser_usage.mjlab.recording import RolloutRecorder
from mujoco_viser_usage.mjlab.tasks import environment_config
from mujoco_viser_usage.rl.storage import read_jsonl


@pytest.mark.parametrize(
    "stage,dimension", [("reach", 14), ("lift", 7), ("place", 7), ("bimanual", 14)]
)
def test_physics_actions_and_terminal_recording(tmp_path, stage, dimension):
    cfg = environment_config(stage, num_envs=2, seed=13)
    cfg.episode_length_s = 0.08
    cfg.recorders = {
        "trace": RecorderTermCfg(
            func=RolloutRecorder, params={"directory": str(tmp_path), "every": 2}
        )
    }
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observation, _ = env.reset(seed=13)
        assert float(env.sim.data.ctrl[0, 1]) == pytest.approx(-0.96)
        assert float(env.sim.data.ctrl[0, 2]) == pytest.approx(1.16)
        qpos = env.sim.data.qpos[:].clone()
        goals = mdp.state(env).goals.clone()
        env.reset(seed=13)
        torch.testing.assert_close(env.sim.data.qpos[:], qpos)
        torch.testing.assert_close(mdp.state(env).goals, goals)
        assert observation["actor"].shape == (2, 66 + dimension)
        action = torch.zeros((2, dimension))
        action[0, 0] = 0.3
        for _ in range(4):
            observation, reward, _terminated, timeout, _ = env.step(action)
            assert torch.isfinite(observation["actor"]).all()
            assert torch.isfinite(reward).all()
        assert timeout.all()
        records = read_jsonl(tmp_path / "workers/0/episodes.jsonl")
        assert records[-1]["state"] == "complete"
        with np.load(tmp_path / "workers/0" / records[-1]["trace"]) as trace:
            assert trace["time"][0] == 0
            assert trace["time"][-1] == pytest.approx(0.08)
            assert np.all(np.diff(trace["time"]) > 0)
            assert trace["action"][-1, 0] == pytest.approx(0.3)
            assert np.linalg.norm(trace["qpos"][-1] - trace["qpos"][0]) > 1e-4
            # A terminal trace must retain its actual terminal pose, not reset pose.
            assert not np.allclose(trace["qpos"][-1], env.sim.data.qpos[0].numpy())
        export_scene(env, tmp_path / "scene")
        exported = mujoco.MjModel.from_xml_path(str(tmp_path / "scene/scene.xml"))
        assert exported.opt.impratio == env.sim.mj_model.opt.impratio
        assert exported.opt.integrator == env.sim.mj_model.opt.integrator
        assert (exported.nq, exported.nu, exported.nmocap) == (
            env.sim.mj_model.nq,
            14,
            1,
        )
    finally:
        env.close()


def test_independent_arm_targets_and_incremental_range():
    env = ManagerBasedRlEnv(environment_config("bimanual", num_envs=2), device="cpu")
    try:
        env.reset()
        term = env.action_manager.get_term("servo")
        action = torch.zeros((2, 14))
        action[0, 0] = 1
        action[1, 7] = -1
        for _ in range(30):
            term.process_actions(action)
        term.apply_actions()
        env.scene.write_data_to_sim()
        ctrl = env.sim.data.ctrl[:]
        assert ctrl[0, 0] == pytest.approx(1.2)
        assert ctrl[0, 7] == pytest.approx(0.0)
        assert ctrl[1, 0] == pytest.approx(0.0)
        assert ctrl[1, 7] == pytest.approx(-1.2)
    finally:
        env.close()


def test_reach_shaping_rewards_completion_instead_of_waiting():
    env = ManagerBasedRlEnv(environment_config("reach"), device="cpu")
    try:
        env.reset()
        mdp.state(env).goals[:] = mdp.tools(env)
        for _ in range(10):
            _, reward, terminal, _, _ = env.step(torch.zeros((1, 14)))
            if terminal[0]:
                assert float(reward[0]) > 1.9
                break
            assert float(reward[0]) <= 0.0
        else:
            pytest.fail("Holding the reachable goal must terminate successfully")
    finally:
        env.close()


def test_go2_residual_trot_records_base_and_feet(tmp_path):
    from mujoco_viser_usage.mjlab import go2

    cfg = environment_config("go2", num_envs=2, seed=5)
    cfg.episode_length_s = 0.08
    cfg.recorders = {
        "trace": RecorderTermCfg(
            func=RolloutRecorder, params={"directory": str(tmp_path), "every": 2}
        )
    }
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observation, _ = env.reset(seed=5)
        assert observation["actor"].shape == (2, 61)
        assert env.sim.mj_model.nu == 12
        qpos = env.sim.data.qpos[:].clone()
        command = go2.state(env).command.clone()
        assert ((command >= 0.2) & (command <= 0.45)).all()
        env.reset(seed=5)
        torch.testing.assert_close(env.sim.data.qpos[:], qpos)
        torch.testing.assert_close(go2.state(env).command, command)
        term = env.action_manager.get_term("servo")
        torch.testing.assert_close(term.targets[0], go2.state(env).home)
        action = torch.zeros((2, 12))
        action[0, 0] = 0.5
        # Residual acts on the servo target of the first actuator only.
        term.process_actions(action)
        term.apply_actions()
        assert float(term.targets[0, 0] - term.nominal()[0, 0]) == pytest.approx(0.075)
        assert float(term.targets[1, 0] - term.nominal()[1, 0]) == pytest.approx(0.0)
        for _ in range(4):
            observation, reward, _terminated, timeout, _ = env.step(action)
            assert torch.isfinite(observation["actor"]).all()
            assert torch.isfinite(reward).all()
        assert timeout.all()
        records = read_jsonl(tmp_path / "workers/0/episodes.jsonl")
        assert records[-1]["state"] == "complete"
        with np.load(tmp_path / "workers/0" / records[-1]["trace"]) as trace:
            assert trace["base_pos"].shape[1:] == (3,)
            assert trace["base_pos"][-1, 2] > 0.2
            assert trace["base_upright"][-1] > 0.9
            assert (
                trace["contact_force"].reshape(len(trace["time"]), -1, 3).shape[1] == 4
            )
            assert trace["command"][-1] == pytest.approx(float(command[0]))
            assert trace["targets"].shape == (len(trace["time"]), 2, 3)
        export_scene(env, tmp_path / "scene")
        exported = mujoco.MjModel.from_xml_path(str(tmp_path / "scene/scene.xml"))
        assert exported.nu == 12
        assert exported.actuator_gainprm[0, 0] == pytest.approx(150)
        assert mujoco.mj_name2id(exported, mujoco.mjtObj.mjOBJ_GEOM, "wall_0_0") >= 0
        assert mujoco.mj_name2id(exported, mujoco.mjtObj.mjOBJ_BODY, "robot/base") >= 0
    finally:
        env.close()


def test_go2_gait_matches_scripted_reference():
    from mujoco_viser_usage.mjlab import go2
    from mujoco_viser_usage.scenes.go2_task import gait_targets as reference

    times = torch.tensor([0.0, 0.37, 1.2, 2.55])
    out = go2.gait_targets(times, torch.full((4,), 0.15))
    for i, time in enumerate(times.tolist()):
        np.testing.assert_allclose(
            out[i].numpy(), reference(time, stride=0.15), atol=1e-5
        )


def test_go2_terrain_variant_copies_mesh_terrain():
    env = ManagerBasedRlEnv(environment_config("go2", terrain="ramps"), device="cpu")
    try:
        env.reset()
        model = env.sim.mj_model
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "terrain_ramp") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    finally:
        env.close()
