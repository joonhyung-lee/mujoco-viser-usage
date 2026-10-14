"""Comparison clock, seek, float32 replay, and measured-force boundary checks."""

import json

import mujoco
import numpy as np
import pytest

pytest.importorskip("gymnasium")
from viser_robotics.rl.comparison import ComparisonViewer


def test_comparison_pause_seek_and_terminal_hold(tmp_path):
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body name="robot" mocap="true"><geom type="sphere" size=".02"/>
        <site name="robot/left/gripper" pos=".1 0 0"/>
        <site name="robot/right/gripper" pos="-.1 0 0"/>
      </body>
      <body><freejoint/><geom type="box" size=".02 .02 .02"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    reports = []
    for controller, duration in [("scripted", 1.0), ("policy", 2.0)]:
        directory = tmp_path / controller
        (directory / "scene").mkdir(parents=True)
        (directory / "scene/scene.xml").write_text(xml)
        worker = directory / "workers/0"
        worker.mkdir(parents=True)
        qpos = np.tile(model.qpos0.astype(np.float32), (3, 1))
        qpos[:, 0] = np.linspace(0, duration, 3)
        np.savez(
            worker / "episode.npz",
            time=np.linspace(0, duration, 3),
            qpos=qpos,
            targets=np.zeros((3, 2, 3)),
            mocap_pos=np.zeros((3, 1, 3)),
            mocap_quat=np.tile([[[1.0, 0, 0, 0]]], (3, 1, 1)),
            contact_pos=np.zeros((3, 2, 1, 3)),
            contact_force=np.zeros((3, 2, 1, 3)),
        )
        reports.append(
            {
                "seed": 7,
                "controller": controller,
                "directory": controller,
                "trace": "episode.npz",
                "events": [{"kind": "timeout", "time": duration}],
                "phases": ["Reach", "Reach"],
                "is_success": False,
                "truncated": True,
                "return": 0.0,
                "final_goal_error_m": 0.1,
                "max_recorded_contact_force_n": 0.0,
                "max_recorded_penetration_m": 0.0,
            }
        )
    (tmp_path / "comparison.json").write_text(json.dumps({"reports": reports}))
    viewer = ComparisonViewer(tmp_path, port=0)
    try:
        assert viewer.overview.manipulation
        viewer.play.value = False
        viewer.seek(0.5)
        viewer.tick(viewer.previous + 10)
        assert viewer.position == 0.5
        np.testing.assert_allclose([data.qpos[0] for data in viewer.data], 0.5)
        viewer.play.value = True
        viewer.speed.value = "2×"
        viewer.tick(viewer.previous + 0.5)
        assert viewer.position == 1.5
        assert viewer.data[0].qpos[0] == 1.0
        assert viewer.data[1].qpos[0] == 1.5
        viewer.seek(0.0)
        assert all(len(overlay.handles) == 0 for overlay in viewer.forces)
        assert all(data.qpos[0] == 0.0 for data in viewer.data)
    finally:
        viewer.close()
