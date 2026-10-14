"""Upload compiled MuJoCo geometry once; stream world poses thereafter."""

import mujoco
import numpy as np
import trimesh
import viser

from .overlays import EgoCamera, ForceOverlay, TrajectoryOverlay
from .scene_style import configure_scene


def add_translucent_frame(server: viser.ViserServer) -> viser.FrameHandle:
    """Mesh children provide opacity while the parent owns pose/visibility."""
    frame = server.scene.add_frame("/tool", show_axes=False)
    shaft = trimesh.creation.cylinder(radius=0.003, height=0.102, sections=12)
    shaft.apply_translation((0, 0, 0.051))
    tip = trimesh.creation.cone(radius=0.007, height=0.018, sections=12)
    tip.apply_translation((0, 0, 0.102))
    arrow = trimesh.util.concatenate((shaft, tip))
    for name, direction, color in (
        ("x", (1, 0, 0), (240, 90, 90)),
        ("y", (0, 1, 0), (115, 220, 125)),
        ("z", (0, 0, 1), (105, 155, 245)),
    ):
        mesh = arrow.copy()
        mesh.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), direction))
        server.scene.add_mesh_simple(
            f"/tool/{name}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=color,
            opacity=0.55,
            cast_shadow=False,
            receive_shadow=False,
        )
    return frame


def geometry_mesh(model: mujoco.MjModel, index: int):
    kind = model.geom_type[index]
    size = model.geom_size[index]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[index]
        v = model.mesh_vertadr[mesh_id]
        f = model.mesh_faceadr[mesh_id]
        return trimesh.Trimesh(
            vertices=model.mesh_vert[v : v + model.mesh_vertnum[mesh_id]].copy(),
            faces=model.mesh_face[f : f + model.mesh_facenum[mesh_id]].copy(),
            process=False,
        )
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        mesh = trimesh.creation.box(
            extents=(2 * max(size[0], 2), 2 * max(size[1], 2), 0.02)
        )
        mesh.apply_translation((0, 0, -0.01))
        return mesh
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_scale(size)
        return mesh
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    raise ValueError(f"Unsupported MuJoCo geometry type: {kind}")


class PhysicsViewer:
    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        *,
        connect_camera: bool = True,
        show_plan: bool = True,
    ):
        self.server = server
        self.handles = {}
        self.collision_ids = []
        server.scene.set_up_direction("+z")
        configure_scene(server)
        for i in range(model.ngeom):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                # Use one grid surface, avoiding coplanar floor meshes and
                # shadow acne. Its pose is synchronized like every other geom.
                self.handles[i] = server.scene.add_grid(
                    f"/geoms/{i}",
                    width=200,
                    height=200,
                    plane="xy",
                    cell_size=0.5 if model.stat.extent > 5 else 0.25,
                    cell_color=(157, 168, 179),
                    cell_thickness=0.6,
                    section_size=2.0 if model.stat.extent > 5 else 1.0,
                    section_color=(123, 137, 150),
                    section_thickness=1.2,
                    infinite_grid=True,
                    fade_distance=max(12, float(model.stat.extent) * 6),
                    fade_strength=2,
                    plane_color=(184, 194, 204),
                    plane_opacity=1.0,
                    shadow_opacity=0.35,
                )
                continue
            mesh = geometry_mesh(model, i)
            material = model.geom_matid[i]
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]
            collision = model.geom_group[i] == 3
            self.handles[i] = server.scene.add_mesh_simple(
                f"/geoms/{i}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=tuple(int(x * 255) for x in rgba[:3]),
                opacity=0.25 if collision else float(rgba[3]),
                wireframe=collision,
                visible=not collision,
                cast_shadow=not collision,
                receive_shadow=False,
            )
            if collision:
                self.collision_ids.append(i)
        self.tool_frame = add_translucent_frame(server)
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.tool_frame.visible = self.site_id >= 0

        grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if grasp_site >= 0:
            self.site_id = grasp_site
            self.tool_frame.visible = True
        self._tabletop = grasp_site >= 0
        for name in ("tracking_site", "left/gripper", "robot/left/gripper"):
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site >= 0:
                self.site_id = site
                self.tool_frame.visible = True
                break
        self._maze = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal") >= 0
        if self._maze and show_plan:
            from .scenes.maze import route

            points = np.column_stack((route(), np.full(len(route()), 0.025)))
            server.scene.add_line_segments(
                "/planned_route",
                points=np.stack((points[:-1], points[1:]), axis=1).astype(np.float32),
                colors=(118, 185, 0),
                line_width=3,
            )
            server.scene.add_label(
                "/goal_label", "GOAL", position=(*points[-1, :2], 0.18)
            )
        self._scene_center = model.stat.center.copy()
        self._show_contacts = False
        self.contact_points = server.scene.add_point_cloud(
            "/contacts",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=(255, 90, 40),
            point_size=0.008,
            visible=False,
        )
        self.forces = ForceOverlay(server, model)
        self.trajectory = TrajectoryOverlay(server, self.site_id)
        self.trajectories = [self.trajectory]
        right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")
        if right_site < 0:
            right_site = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "robot/right/gripper"
            )
        if right_site >= 0:
            self.trajectories.append(
                TrajectoryOverlay(server, right_site, "/trajectory_right")
            )
        self.ego = EgoCamera(server, model)
        if connect_camera:
            server.on_client_connect(self.reset_camera)

    def reset_camera(self, client) -> None:
        if client is None:
            return
        if self._maze:
            target = self._scene_center
            position = target + (16.0, 14.0, 16.0)
        elif self._tabletop:
            target, position = (0.35, 0.0, 0.8), (2.2, 1.9, 1.8)
        else:
            target = self._scene_center
            position = target + (1.8, 1.8, 1.5)
        # Position first: Viser translates the orbit target when position changes.
        client.camera.position = position
        client.camera.look_at = target
        client.camera.fov = np.deg2rad(45)

    def show_tool_frame(self, visible: bool) -> None:
        self.tool_frame.visible = visible and self.site_id >= 0

    def close(self) -> None:
        self.ego.close()

    def show_forces(self, visible: bool) -> None:
        self.forces.enabled = visible

    def show_trajectory(self, visible: bool) -> None:
        for trajectory in self.trajectories:
            trajectory.enabled = visible

    def show_ego(self, visible: bool) -> None:
        self.ego.enabled = visible

    def show_contacts(self, visible: bool) -> None:
        self._show_contacts = visible
        self.contact_points.visible = visible

    def show_collisions(self, visible: bool):
        for i in self.collision_ids:
            self.handles[i].visible = visible

    def update(self, data: mujoco.MjData):
        with self.server.atomic():
            self.forces.update(data)
            for trajectory in self.trajectories:
                trajectory.update(data)
            self.contact_points.visible = self._show_contacts and data.ncon > 0
            if self._show_contacts and data.ncon:
                self.contact_points.points = data.contact.pos.copy().astype(np.float32)
            for i, handle in self.handles.items():
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, data.geom_xmat[i])
                handle.position = data.geom_xpos[i].copy()
                handle.wxyz = quat
            if self.site_id >= 0:
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
                self.tool_frame.position = data.site_xpos[self.site_id].copy()
                self.tool_frame.wxyz = quat

        self.ego.update(data)
