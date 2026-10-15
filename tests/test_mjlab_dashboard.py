"""Cross-interpreter lifecycle and model replay for the native dashboard backend."""

import importlib.util
import time
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from mujoco_viser_usage.rl.config import MjlabTrainConfig, parse_config
from mujoco_viser_usage.rl.dashboard import ControlRoom
from mujoco_viser_usage.rl.storage import read_json
from mujoco_viser_usage.rl.supervisor import RunSupervisor

HAS_MJLAB = importlib.util.find_spec("mjlab") is not None


def await_state(supervisor, run, predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor.poll()
        status = read_json(run / "status.json", {})
        if predicate(status):
            return status
        if status.get("state") == "failed":
            pytest.fail((run / "process.log").read_text()[-5000:])
        time.sleep(0.05)
    pytest.fail(f"Timed out: {read_json(run / 'status.json')}")


def test_native_request_roundtrip_and_invalid_task():
    config = MjlabTrainConfig(workers=512, device="cuda:0")
    assert parse_config(config.to_dict()) == config
    with pytest.raises(ValueError):
        MjlabTrainConfig(task="aloha_reach", terrain="ramps")
    assert MjlabTrainConfig(task="go2_velocity", terrain="mixed").terrain == "mixed"


@pytest.mark.skipif(not HAS_MJLAB, reason="uv sync --extra mjlab required")
def test_native_pause_stop_resume_and_dashboard_replay(tmp_path):
    supervisor = RunSupervisor(tmp_path)
    config = MjlabTrainConfig(workers=2, n_steps=8, total_timesteps=4096)
    room = None
    try:
        run = supervisor.start(config)
        await_state(supervisor, run, lambda state: state.get("run_steps", 0) >= 16)
        supervisor.command(run, "pause")
        paused = await_state(
            supervisor, run, lambda state: state.get("state") == "paused"
        )
        time.sleep(0.25)
        assert read_json(run / "status.json")["run_steps"] == paused["run_steps"]
        supervisor.command(run, "run")
        await_state(
            supervisor,
            run,
            lambda state: state.get("run_steps", 0) > paused["run_steps"],
        )
        supervisor.command(run, "stop")
        await_state(supervisor, run, lambda state: state.get("state") == "stopped")
        supervisor.jobs[run].wait(timeout=10)
        assert (run / "stopped.pt").exists()
        assert read_json(run / "workers/0/state.json")["state"] == "closed"
        resumed = supervisor.start(
            replace(config, total_timesteps=32), resume=run / "stopped.pt"
        )
        status = await_state(
            supervisor, resumed, lambda state: state.get("state") == "completed"
        )
        supervisor.jobs[resumed].wait(timeout=10)
        assert status["run_steps"] == 32
        assert (resumed / "final.pt").exists()
        assert not list(resumed.glob("*.tmp"))
        comparison = supervisor.compare(resumed / "final.pt")
        await_state(
            supervisor, comparison, lambda state: state.get("state") == "running"
        )
        supervisor.command(comparison, "stop")
        await_state(
            supervisor, comparison, lambda state: state.get("state") == "stopped"
        )
        supervisor.jobs[comparison].wait(timeout=10)
        assert not (comparison / "comparison.json").exists()
        room = ControlRoom(tmp_path, port=0)
        room._select_run(resumed.name)
        assert room.backend.value == "mjlab"
        assert room.loaded_model_path == resumed / "scene/scene.xml"
        assert room.sim.model.nmocap == 1
        assert len(room.viewer.trajectories) == 2
        room._refresh_status()
        assert room._checkpoint_path() == resumed / "final.pt"
        with np.load(resumed / "workers/0/latest.npz") as archive:
            frame = dict(archive)
        room._apply_frame(frame)
        np.testing.assert_allclose(room.sim.data.qpos, frame["qpos"])
        np.testing.assert_allclose(room.sim.data.mocap_pos, frame["mocap_pos"])
        room.view_mode.value = "Parallel overview"
        room._change_view()
        room._refresh_rollouts()
        assert room.overview.manipulation
        assert len(room.overview.trails[0]) == 2
    finally:
        if room:
            room.close()
        supervisor.close()


@pytest.mark.skipif(not HAS_MJLAB, reason="uv sync --extra mjlab required")
def test_native_go2_run_replays_in_dashboard(tmp_path):
    supervisor = RunSupervisor(tmp_path)
    config = MjlabTrainConfig(
        task="go2_velocity", terrain="steps", workers=2, n_steps=8, total_timesteps=32
    )
    room = None
    try:
        run = supervisor.start(config)
        status = await_state(
            supervisor, run, lambda state: state.get("state") == "completed"
        )
        supervisor.jobs[run].wait(timeout=10)
        assert status["run_steps"] == 32
        assert read_json(run / "run.json")["stage"] == "go2"
        room = ControlRoom(tmp_path, port=0)
        room._select_run(run.name)
        assert room.backend.value == "mjlab"
        assert room.terrain_choice.value == "steps"
        assert room.loaded_model_path == run / "scene/scene.xml"
        assert room.base_body == "robot/base"
        assert room.sim.model.nmocap == 0
        with np.load(run / "workers/0/latest.npz") as archive:
            frame = dict(archive)
        assert "mocap_pos" not in frame
        room._apply_frame(frame)
        np.testing.assert_allclose(room.sim.data.qpos, frame["qpos"])
        for client in room.server.get_clients().values():
            room._focus(client)
        room.view_mode.value = "Parallel overview"
        room._change_view()
        room._refresh_rollouts()
        assert not room.overview.manipulation
    finally:
        if room:
            room.close()
        supervisor.close()
