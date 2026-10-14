"""Terrain selection changes contact geometry in the actual RL environment."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from mujoco_viser_usage.rl.config import TrainConfig
from mujoco_viser_usage.rl.envs import RobotEnv
from mujoco_viser_usage.scenes import TERRAINS, resolve_scene
from mujoco_viser_usage.viser_parser import ViserParser


@pytest.mark.parametrize("terrain", TERRAINS)
def test_rl_terrain_physics(terrain):
    env = RobotEnv(TrainConfig(terrain=terrain, max_episode_steps=150))
    obs, _ = env.reset(seed=42)
    ids = [
        i
        for i in range(env.model.ngeom)
        if env.model.geom(i).name.startswith("terrain_")
    ]
    assert bool(ids) == (terrain != "flat")
    assert all(env.model.geom_contype[i] for i in ids)
    for _ in range(150):
        obs, reward, done, truncated, _info = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not np.any(env.data.warning.number)
        if done or truncated:
            break
    env.close()


@pytest.mark.parametrize("terrain", TERRAINS[1:])
def test_real_husky_drives_over_terrain(terrain):
    with ViserParser.from_scene(
        "navigation/husky_" + terrain, viewer=False, demo=True
    ) as robot:
        assert robot.model.nmesh == (6 if terrain in {"ramps", "mixed"} else 5)
        for _ in range(round(180 / robot.model.opt.timestep / 100)):
            robot.step(nstep=100)
            assert not np.any(robot.data.warning.number)
            if not robot.task.active:
                break
        assert robot.task.status == "Goal reached"
        assert robot.data.body("base").xmat[8] > 0.9


def test_old_config_defaults_to_original_terrain():
    assert TrainConfig().terrain == "flat"
    with pytest.raises(ValueError, match="only to Go2"):
        TrainConfig(task="aloha_reach", terrain="mixed")
    assert resolve_scene("navigation/husky").path.exists()
