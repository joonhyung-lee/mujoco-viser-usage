"""Scripted Franka plug insertion and cable routing complete through contact."""

import pytest

from mujoco_viser_usage import ViserParser


@pytest.mark.parametrize("name", ["plug", "cable"])
def test_franka_task_completes(name):
    with ViserParser.from_scene(f"manipulation/franka_{name}", viewer=False) as robot:
        robot.task.start()
        penetration = 0.0
        for _ in range(12000):
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
