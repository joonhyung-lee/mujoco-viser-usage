import numpy as np

from viser_robotics import ViserParser


def test_default_runs_and_paused_option():
    with ViserParser.from_scene("locomotion/go2", viewer=False) as robot:
        assert robot.runtime.running
        robot.runtime.tick(0.02)  # Discard initialization wall time.
        robot.runtime.tick(0.02)
        assert robot.data.time > 0
    with ViserParser.from_scene(
        "locomotion/go2", viewer=False, start_running=False
    ) as robot:
        robot.runtime.tick(0.1)
        robot.runtime.tick(0.1)
        assert robot.data.time == 0


def test_aloha_places_both_objects_with_bounded_contact_depth():
    with ViserParser.from_scene("manipulation/aloha", viewer=False, demo=True) as robot:
        maximum_height = np.zeros(2)
        depth = 0.0
        for _ in range(22000):
            robot.step()
            maximum_height = np.maximum(
                maximum_height,
                [robot.data.body(f"object_{i}").xpos[2] for i in range(2)],
            )
            depth = max(depth, -min(robot.data.contact.dist, default=0))
            if not robot.task.active:
                break
        assert robot.task.status.startswith("Complete")
        assert np.all(maximum_height > 0.13)
        assert depth < 0.001
        for i in range(2):
            np.testing.assert_allclose(
                robot.data.body(f"object_{i}").xpos, robot.task.targets[i], atol=0.008
            )
        robot.reset()
        assert not robot.task.active


def test_go2_trot_moves_using_physics_and_stops():
    with ViserParser.from_scene("locomotion/go2", viewer=False, demo=True) as robot:
        contacts = 0
        for _ in range(7000):
            robot.step()
            contacts += robot.data.ncon
            assert np.isfinite(robot.data.qpos).all()
            if not robot.task.active:
                break
        assert robot.task.status.startswith("Complete")
        assert robot.data.body("base").xpos[0] > 1.8
        assert abs(robot.data.body("base").xpos[1]) < 0.3
        assert robot.data.body("base").xmat[8] > 0.9
        assert contacts > 0
        robot.step(nstep=1000)
        assert robot.data.body("base").xpos[2] > 0.2
