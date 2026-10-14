"""Collision meshes must also render as outward-facing closed surfaces."""

from viser_robotics.scenes import resolve_scene
from viser_robotics.simulation import Simulation
from viser_robotics.viewer import geometry_mesh


def test_ramp_faces_point_outward():
    sim = Simulation(resolve_scene("locomotion/go2_mixed").path)
    mesh = geometry_mesh(sim.model, sim.model.geom("terrain_ramp").id)
    assert mesh.is_watertight
    assert mesh.volume > 0, "Inward ramp faces disappear with backface culling"
