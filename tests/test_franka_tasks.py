"""Scripted Franka plug insertion and cable routing complete through contact."""

import pytest

from mujoco_viser_usage import ViserParser


@pytest.mark.parametrize(
    "scene",
    [
        "manipulation/franka_plug",
        "manipulation/franka_cable",
        "manipulation/nist_board",
    ],
)
def test_franka_task_completes(scene):
    with ViserParser.from_scene(scene, viewer=False) as robot:
        robot.task.start()
        penetration = 0.0
        for _ in range(60000):
            robot.step()
            penetration = max(
                penetration, -float(min(robot.data.contact.dist, default=0))
            )
            if not robot.task.active:
                break
        assert robot.task.status.startswith("Complete"), robot.task.status
        assert penetration < 0.001, f"{penetration * 1000:.2f} mm"
        robot.reset()
        assert not robot.task.active
