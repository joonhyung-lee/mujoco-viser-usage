"""Scene selection must resolve matching run, model and recorded worker state."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from mujoco_viser_usage.rl.config import TrainConfig, scene_path
from mujoco_viser_usage.rl.dashboard import ControlRoom
from mujoco_viser_usage.rl.storage import write_json, write_npz
from mujoco_viser_usage.simulation import Simulation


def make_run(root: Path, task: str):
    run = root / task
    config = TrainConfig(task=task, workers=1)
    write_json(run / "config.json", config.to_dict())
    write_json(run / "status.json", {"state": "completed"})
    sim = Simulation(scene_path(config))
    write_npz(
        run / "workers/0/episode-000000.npz",
        time=np.array([0.0, 1.0]),
        qpos=np.tile(sim.data.qpos, (2, 1)),
        qvel=np.tile(sim.data.qvel, (2, 1)),
        ctrl=np.tile(sim.data.ctrl, (2, 1)),
        targets=np.zeros((2, 2, 3)),
    )
    return run


def test_scene_switch_uses_matching_model_and_trace(tmp_path):
    aloha = make_run(tmp_path, "aloha_pick_place")
    go2 = make_run(tmp_path, "go2_velocity")
    room = ControlRoom(tmp_path, port=0)
    try:
        assert room.selected_run == aloha
        assert room.sim.model.nu == 14
        room._refresh_frame()
        assert room.players[0].path.parent.parent.parent == aloha
        room._preview_task("Go2 · walk")
        assert room.selected_run == go2
        assert room.sim.model.nu == 12
        room._refresh_frame()
        assert room.players[0].path.parent.parent.parent == go2
        room._preview_task("ALOHA · pick & place")
        assert room.selected_run == aloha
        room.view_mode.value = "Parallel overview"
        room._change_view()
        room._refresh_frame()
        assert len(room.overview.roots) == 1
        assert len(room.overview.trails[0]) == 2
        assert len(room.overview.goals[0]) == 2
    finally:
        room.close()
