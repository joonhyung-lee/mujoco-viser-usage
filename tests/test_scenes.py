"""Catalog scenes must load independently and exercise real contact dynamics."""

import mujoco
import numpy as np
import pytest

from mujoco_viser_usage.scenes import SCENES, resolve_scene
from mujoco_viser_usage.scenes.maze import route
from mujoco_viser_usage.viser_parser import ViserParser


@pytest.mark.parametrize("scene", SCENES, ids=lambda scene: scene.id)
def test_catalog_physics(scene):
    with ViserParser(
        scene.path, viewer=False, pick_place=scene.task == "pick_place"
    ) as robot:
        initial = robot.data.qpos.copy()
        assert min(robot.data.contact.dist, default=0) > -0.001
        robot.step(nstep=round(4 / robot.model.opt.timestep))
        assert np.isfinite(robot.data.qpos).all()
        assert not np.any(robot.data.warning.number)
        assert min(robot.data.contact.dist, default=0) > -0.001
        if scene.id.startswith("locomotion/"):
            # Floating root remains upright and inside its starting cell.
            assert robot.data.qpos[2] > 0.2
            quat = robot.data.qpos[3:7]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, quat)
            assert matrix[8] > 0.9
            assert np.linalg.norm(robot.data.qpos[:2]) < 0.4
        robot.reset()
        np.testing.assert_allclose(robot.data.qpos, initial)


@pytest.mark.parametrize("name", ["husky", "turtlebot3"])
def test_wheels_reach_maze_goal(name):
    scene = resolve_scene("navigation/" + name)
    with ViserParser(scene.path, viewer=False, navigation=True) as robot:
        robot.task.start()
        worst_wall_depth = 0.0
        for _ in range(900):
            robot.step(nstep=100)
            assert np.isfinite(robot.data.qpos).all()
            for contact in robot.data.contact:
                if any(
                    robot.model.geom(g).name.startswith("wall_")
                    for g in (contact.geom1, contact.geom2)
                ):
                    worst_wall_depth = max(worst_wall_depth, -contact.dist)
            if not robot.task.active:
                break
        assert robot.task.status == "Goal reached"
        assert np.linalg.norm(robot.data.body("base").xpos[:2] - route()[-1]) < 0.14
        assert worst_wall_depth < 0.001
        np.testing.assert_array_equal(robot.data.ctrl, 0.0)
        robot.reset()
        assert not robot.task.active
        np.testing.assert_allclose(robot.data.body("base").xpos[:2], [0, 0])
