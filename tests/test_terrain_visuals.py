"""Collision meshes must also render as outward-facing closed surfaces."""

from mujoco_viser_usage.scenes import resolve_scene
from mujoco_viser_usage.simulation import Simulation
from mujoco_viser_usage.viewer import raw_geometry_mesh


def test_ramp_faces_point_outward():
    sim = Simulation(resolve_scene("locomotion/go2_mixed").path)
    mesh = raw_geometry_mesh(sim.model, sim.model.geom("terrain_ramp").id)
    assert mesh.is_watertight
    assert mesh.volume > 0, "Inward ramp faces disappear with backface culling"
