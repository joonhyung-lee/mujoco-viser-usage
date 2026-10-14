import numpy as np
import pytest

from mujoco_viser_usage import ViserParser
from mujoco_viser_usage.simulation import PICK_PLACE_SCENE


def test_contact_grasp_lift_transfer_release_and_reset():
    with ViserParser(PICK_PLACE_SCENE, viewer=False, pick_place=True) as robot:
        initial = robot.get_q()
        robot.task.start()
        max_height = 0.0
        finger_contact = False
        maximum_penetration = 0.0
        for _ in range(10500):
            robot.step()
            max_height = max(max_height, robot.data.body("pick_object").xpos[2])
            for contact in robot.data.contact:
                maximum_penetration = max(maximum_penetration, -float(contact.dist))
                names = [
                    robot.model.geom(int(g)).name
                    for g in (contact.geom1, contact.geom2)
                ]
                if "pick_object_geom" in names and any(
                    "finger" in name for name in names
                ):
                    finger_contact = True
            if not robot.task.active:
                break
        assert robot.task.status == "Complete"
        assert maximum_penetration < 0.001, (
            f"Penetration: {maximum_penetration * 1000:.3f} mm"
        )
        assert finger_contact
        assert max_height > 0.93
        pos = robot.data.body("pick_object").xpos
        assert np.linalg.norm(pos[:2] - [0.65, 0.22]) < 0.04
        assert pos[2] == pytest.approx(0.775, abs=0.005)
        assert robot.data.ctrl[-1] == pytest.approx(1)
        robot.reset()
        assert robot.task.status == "Ready"
        assert not robot.task.active
        np.testing.assert_allclose(robot.get_q(), initial)
        robot.task.start()
        robot.step(nstep=100)
        robot.reset()
        robot.step(nstep=10)
        assert not robot.task.active
        assert robot.task.status == "Ready"
