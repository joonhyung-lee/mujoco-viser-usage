import numpy as np
import pytest

from viser_robotics.simulation import Simulation


def test_motor_target_drives_physics_without_teleporting():
    sim = Simulation()
    initial = sim.data.qpos.copy()
    target = sim.data.ctrl[0] + 0.25
    sim.set_control(0, target)
    np.testing.assert_array_equal(sim.data.qpos, initial)
    sim.step(1500)
    assert abs(sim.data.qpos[0] - target) < 0.02
    assert np.isfinite(sim.data.qpos).all()
    assert sim.data.time == pytest.approx(3.0)


def test_gravity_contact_drop_and_reset():
    sim = Simulation()
    initial = sim.data.qpos.copy()
    sim.step(100)
    assert sim.data.body("drop_box").xpos[2] < 0.65
    sim.step(1400)
    assert sim.data.body("drop_box").xpos[2] == pytest.approx(0.045, abs=0.002)
    assert sim.data.ncon > 0
    sim.drop_box()
    assert sim.data.body("drop_box").xpos[2] == pytest.approx(0.65)
    sim.reset()
    assert sim.data.time == 0
    np.testing.assert_array_equal(sim.data.qpos, initial)
    np.testing.assert_array_equal(sim.data.qvel, 0)


def test_control_validation():
    sim = Simulation()
    sim.set_control(0, 100)
    assert sim.data.ctrl[0] == sim.model.actuator_ctrlrange[0, 1]
    with pytest.raises(ValueError):
        sim.set_control(0, float("nan"))
