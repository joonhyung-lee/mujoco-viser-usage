"""Regression tests at the dashboard seam and episode boundary."""

from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from viser_robotics.rl.dashboard import ControlRoom
from viser_robotics.rl.playback import PlaybackClock, RolloutPlayer, sample_trace
from viser_robotics.rl.storage import append_json, write_npz
from viser_robotics.scenes import resolve_scene
from viser_robotics.simulation import Simulation


def trace_fixture():
    sim = Simulation(resolve_scene("locomotion/go2").path)
    trace = {"time": np.arange(0, 1.01, 0.1), "qpos": np.tile(sim.data.qpos, (11, 1))}
    trace["qpos"][:, 0] = trace["time"]
    return sim, trace


def test_dashboard_replay_has_no_recording_interval_stutter():
    room = ControlRoom.__new__(ControlRoom)
    room.clock = PlaybackClock()
    room.progress = SimpleNamespace(value=0)
    room.loop_rollout = SimpleNamespace(value=False)
    room.overview = None
    room.sim, room.replay = trace_fixture()
    room.replay_started = 0
    room.replay_speed = SimpleNamespace(value="1")
    room.notice = SimpleNamespace(content="")
    seen = []
    room._apply_frame = lambda frame: seen.append(float(frame["qpos"][0]))
    for tick in range(25):
        with patch(
            "viser_robotics.rl.dashboard.time.monotonic", return_value=tick / 30
        ):
            room._refresh_frame()
    np.testing.assert_allclose(np.diff(seen), 1 / 30, atol=1e-12)


def test_quaternion_interpolation_is_normalized_and_trace_is_immutable():
    sim, trace = trace_fixture()
    trace["qpos"][1:, 3:7] = [0, 0, 0, 1]
    original = trace["qpos"].copy()
    mid = sample_trace(sim.model, trace, 0.05)
    np.testing.assert_allclose(mid["qpos"][3:7], [2**-0.5, 0, 0, 2**-0.5], atol=1e-12)
    np.testing.assert_array_equal(trace["qpos"], original)
    np.testing.assert_array_equal(
        sample_trace(sim.model, trace, 20)["qpos"], original[-1]
    )
    assert np.isfinite(mid["qpos"]).all()
    sim.data.qpos[:] = mid["qpos"]
    mujoco.mj_forward(sim.model, sim.data)


def test_buffer_does_not_jump_to_new_episode_mid_rollout(tmp_path):
    sim, trace = trace_fixture()
    player = RolloutPlayer(sim.model)
    assert player.sample(tmp_path, 0) is None
    write_npz(tmp_path / "episode-000001.npz", **trace)
    assert player.sample(tmp_path, 1)["qpos"][0] == 0
    trace["qpos"][:, 0] += 100
    write_npz(tmp_path / "episode-000002.npz", **trace)
    assert player.sample(tmp_path, 1.5)["qpos"][0] == 0.5
    assert player.sample(tmp_path, 2)["qpos"][0] == 1
    assert player.sample(tmp_path, 2.01)["qpos"][0] == 100


def test_pause_and_speed_preserve_playback_position():
    clock = PlaybackClock()
    assert clock.advance(10) == 0
    assert clock.advance(11) == 1
    clock.paused = True
    assert clock.advance(20) == 1
    clock.paused = False
    assert clock.advance(21, 2) == 3
    assert clock.advance(22, 0.5) == 3.5


def test_finished_rollout_repeats_without_interpolating_reset(tmp_path):
    sim, trace = trace_fixture()
    write_npz(tmp_path / "episode-000001.npz", **trace)
    player = RolloutPlayer(sim.model)
    assert player.sample(tmp_path, 0, loop=True)["qpos"][0] == 0
    assert player.sample(tmp_path, 1, loop=True)["qpos"][0] == 1
    assert player.sample(tmp_path, 1.7, loop=True)["qpos"][0] == 0


def test_finished_episode_preferred_over_interrupted_fragment(tmp_path):
    sim, trace = trace_fixture()
    for i, state in ((1, "complete"), (2, "interrupted")):
        name = f"episode-{i:06d}.npz"
        write_npz(tmp_path / name, **trace)
        append_json(tmp_path / "episodes.jsonl", {"state": state, "trace": name})
    player = RolloutPlayer(sim.model)
    player.sample(tmp_path, 0)
    assert player.path.name == "episode-000001.npz"


def test_warp_float32_trace_interpolates_in_native_mujoco():
    sim, trace = trace_fixture()
    trace["qpos"] = trace["qpos"].astype(np.float32)
    sample = sample_trace(sim.model, trace, 0.05)
    assert sample["qpos"].dtype == np.float64
    assert sample["qpos"][0] == pytest.approx(0.05)
    assert trace["qpos"].dtype == np.float32
