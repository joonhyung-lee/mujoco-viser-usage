import numpy as np
import pytest

from viser_robotics import ViserParser
from viser_robotics.runtime import SimulationRuntime
from viser_robotics.simulation import Simulation


def test_headless_parser_physics_and_kinematic_interface():
    with ViserParser(viewer=False) as robot:
        initial = robot.get_q()
        copied = robot.get_q()
        copied[0] = 100
        np.testing.assert_array_equal(robot.get_q(), initial)
        robot.set_control("shoulder_pan", initial[0] + 0.2)
        robot.step(nstep=1000)
        assert robot.data.time == pytest.approx(2)
        assert robot.get_q()[0] == pytest.approx(initial[0] + 0.2, abs=0.02)
        robot.forward(initial)
        assert robot.data.time == pytest.approx(2)
        position, rotation = robot.get_body_pose("wrist_3_link")
        assert position.shape == (3,)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        robot.reset()
        assert robot.data.time == 0
        np.testing.assert_array_equal(robot.get_q(), initial)
        with pytest.raises(RuntimeError):
            robot.run()


def test_runtime_pause_single_step_resume_reset_and_queue():
    sim = Simulation()
    runtime = SimulationRuntime(sim)
    runtime.tick(1)
    assert sim.data.time == 0
    runtime.submit(runtime.single_step)
    assert sim.data.time == 0
    runtime.tick(0)
    assert sim.data.time == pytest.approx(0.002)
    runtime.submit(runtime.toggle)
    runtime.tick(10)  # Time spent paused must not become simulation time.
    assert sim.data.time == pytest.approx(0.002)
    runtime.tick(0.02)
    assert sim.data.time == pytest.approx(0.022)
    runtime.submit(lambda: sim.set_control(0, -1.2))
    runtime.tick(0.02)  # Dragging a slider must not discard the physics budget.
    assert sim.data.time == pytest.approx(0.042)
    runtime.submit(runtime.reset)
    runtime.tick(0.05)
    assert sim.data.time == 0
    assert not runtime.running
    runtime.tick(0.1)
    assert sim.data.time == 0


def test_validation_does_not_partially_write_controls():
    with ViserParser(viewer=False) as robot:
        before = robot.data.ctrl.copy()
        with pytest.raises(ValueError):
            robot.step([1, 2, 3, 4, 5, np.nan])
        np.testing.assert_array_equal(robot.data.ctrl, before)
        for count in [0, -1, 0.5, True]:
            with pytest.raises(ValueError):
                robot.step(nstep=count)


def test_custom_xml_without_robot_or_actuators(tmp_path):
    path = tmp_path / "falling.xml"
    path.write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    with ViserParser(path, viewer=False) as robot:
        robot.step(nstep=10)
        assert robot.data.qpos[2] < 1
        robot.reset()
        assert robot.data.qpos[2] == 1
