import mujoco
import numpy as np

from mujoco_viser_usage.overlays import contact_vectors, path_anchors, tube_mesh


def test_force_arrows_use_world_frame_and_physical_newtons():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
        </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=500)
    mujoco.mj_forward(model, data)
    positions, forces = contact_vectors(model, data)
    assert positions.shape == (1, 3)
    np.testing.assert_allclose(forces.sum(axis=0), [0, 0, 9.81], atol=0.01)


def test_ten_anchors_are_evenly_spaced_by_distance_not_sample_index():
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0, 0], [0.9, 0, 0]])
    anchors = path_anchors(points)
    assert anchors.shape == (10, 3)
    np.testing.assert_allclose(anchors[:, 0], np.linspace(0, 0.9, 10))
    vertices, faces = tube_mesh(anchors)
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0 and faces.max() < len(vertices)
