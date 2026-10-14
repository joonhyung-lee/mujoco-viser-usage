import numpy as np

from mujoco_viser_usage.simulation import PICK_PLACE_SCENE, Simulation


def test_home_ego_camera_is_upright_and_looks_toward_workspace():
    sim = Simulation(PICK_PLACE_SCENE)
    camera = sim.data.camera("ego_rgb")
    rotation = camera.xmat.reshape(3, 3)
    # Positive screen-up projection onto world-up: no inverted wrist image.
    assert rotation[2, 1] > 0.5
    assert -rotation[2, 2] < -0.7
    to_object = sim.data.body("pick_object").xpos - camera.xpos
    assert np.dot(-rotation[:, 2], to_object) > 0
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-8)
