"""Read-only diagnostics: physical contact forces, measured EEF path and RGB."""

import time

import mujoco
import numpy as np
import trimesh
import viser


def contact_vectors(model: mujoco.MjModel, data: mujoco.MjData):
    """Return world-frame forces acting on geom2, including tangential friction."""
    positions, forces = [], []
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact):
        mujoco.mj_contactForce(model, data, index, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        if np.linalg.norm(force) > 0.02:
            positions.append(contact.pos.copy())
            forces.append(force)
    return np.asarray(positions).reshape(-1, 3), np.asarray(forces).reshape(-1, 3)


def path_anchors(points: np.ndarray, count: int = 10) -> np.ndarray:
    """Interpolate samples at equal arc-length intervals, including both ends."""
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-9]
    distance, points = distance[keep], points[keep]
    samples = np.linspace(0, distance[-1], count)
    return np.column_stack(
        [np.interp(samples, distance, points[:, axis]) for axis in range(3)]
    )


def tube_mesh(points: np.ndarray, radius: float = 0.006):
    """An eight-sided translucent tube follows the measured 3D path."""
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    reference = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    reference[np.abs(tangent[:, 2]) > 0.9] = [0, 1, 0]
    normal = np.cross(tangent, reference)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    binormal = np.cross(tangent, normal)
    angles = np.arange(8) * 2 * np.pi / 8
    vertices = (
        points[:, None, :]
        + radius
        * (
            np.cos(angles)[None, :, None] * normal[:, None, :]
            + np.sin(angles)[None, :, None] * binormal[:, None, :]
        )
    ).reshape(-1, 3)
    faces = []
    for ring in range(len(points) - 1):
        for j in range(8):
            a, b = ring * 8 + j, ring * 8 + (j + 1) % 8
            faces.extend(((a, b, a + 8), (b, b + 8, a + 8)))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.uint32)


class ForceOverlay:
    def __init__(
        self, server: viser.ViserServer, model: mujoco.MjModel, *, namespace="/forces"
    ):
        self.server, self.model = server, model
        self.namespace = namespace
        self.enabled = True
        self.handles = []
        shaft = trimesh.creation.cylinder(radius=1, height=0.8, sections=10)
        shaft.apply_translation([0, 0, 0.4])
        tip = trimesh.creation.cone(radius=2.5, height=0.2, sections=10)
        tip.apply_translation([0, 0, 0.8])
        self.mesh = trimesh.util.concatenate([shaft, tip])

    def update(self, data: mujoco.MjData) -> None:
        positions, forces = contact_vectors(self.model, data)
        self.update_vectors(positions, forces)

    def update_vectors(self, positions, forces):
        """Display measured vectors, including forces saved by a different backend."""
        positions = np.asarray(positions).reshape(-1, 3)
        forces = np.asarray(forces).reshape(-1, 3)
        valid = (
            np.isfinite(positions).all(axis=1)
            & np.isfinite(forces).all(axis=1)
            & (np.linalg.norm(forces, axis=1) > 0.02)
        )
        positions, forces = positions[valid], forces[valid]
        indices = (
            np.argsort(np.linalg.norm(forces, axis=1))[::-1][:32]
            if self.enabled
            else []
        )
        while len(self.handles) < len(indices):
            self.handles.append(
                self.server.scene.add_mesh_simple(
                    f"{self.namespace}/{len(self.handles)}",
                    vertices=self.mesh.vertices,
                    faces=self.mesh.faces,
                    color=(255, 160, 65),
                    opacity=0.85,
                    cast_shadow=False,
                    receive_shadow=False,
                )
            )
        for slot, handle in enumerate(self.handles):
            handle.visible = slot < len(indices)
            if not handle.visible:
                continue
            index = indices[slot]
            magnitude = float(np.linalg.norm(forces[index]))
            rotation = trimesh.geometry.align_vectors(
                [0, 0, 1], forces[index] / magnitude
            )[:3, :3]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
            handle.position = positions[index]
            handle.wxyz = quat
            handle.scale = (0.0015, 0.0015, min(magnitude * 0.02, 0.25))


class TrajectoryOverlay:
    def __init__(
        self, server: viser.ViserServer, site_id: int, namespace: str = "/trajectory"
    ):
        self.server, self.site_id = server, site_id
        self.namespace = namespace
        self.root = server.scene.add_frame(namespace, show_axes=False)
        self.enabled = True
        self.points: list[np.ndarray] = []
        self.last_time = 0.0
        self.last_draw = -1.0
        self.lane = None
        self.markers = []
        fill = trimesh.creation.icosphere(subdivisions=2, radius=0.008)
        outline = trimesh.creation.icosphere(subdivisions=2, radius=0.0105)
        for i in range(10):
            frame = server.scene.add_frame(
                f"{self.namespace}/anchor_{i}", show_axes=False, visible=False
            )
            for suffix, mesh, color, side in [
                ("outline", outline, (0, 0, 0), "back"),
                ("fill", fill, (208, 232, 246), "front"),
            ]:
                server.scene.add_mesh_simple(
                    f"{self.namespace}/anchor_{i}/{suffix}",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    color=color,
                    side=side,
                    cast_shadow=False,
                    receive_shadow=False,
                )
            self.markers.append(frame)

    def reset(self) -> None:
        self.points.clear()
        self.last_time = 0.0
        self.last_draw = -1.0
        if self.lane is not None:
            self.lane.remove()
            self.lane = None
        for marker in self.markers:
            marker.visible = False

    def set_history(self, points) -> None:
        """Restore the recorded path preceding a replay seek, without future points."""
        self.reset()
        for point in np.asarray(points)[-1500:]:
            if not self.points or np.linalg.norm(point - self.points[-1]) > 0.002:
                self.points.append(point.copy())

    def update(self, data: mujoco.MjData) -> None:
        self.root.visible = self.enabled and self.site_id >= 0
        if self.site_id < 0:
            return
        if data.time < self.last_time or data.time == 0:
            self.reset()
        self.last_time = float(data.time)
        position = data.site_xpos[self.site_id].copy()
        if not self.points or np.linalg.norm(position - self.points[-1]) > 0.002:
            self.points.append(position)
            self.points = self.points[-1500:]
        if len(self.points) < 2 or (
            self.last_draw >= 0 and data.time - self.last_draw < 0.15
        ):
            return
        self.last_draw = float(data.time)
        points = np.asarray(self.points)
        vertices, faces = tube_mesh(points)
        if self.lane is None:
            self.lane = self.server.scene.add_mesh_simple(
                f"{self.namespace}/lane",
                vertices=vertices,
                faces=faces,
                color=(90, 190, 240),
                opacity=0.28,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )
        else:
            self.lane.vertices = vertices
            self.lane.faces = faces
        for marker, position in zip(self.markers, path_anchors(points)):
            marker.position = position
            marker.visible = True


class EgoCamera:
    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel):
        self.enabled = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_rgb") >= 0
        )
        self.renderer = None
        self.last_render = -1.0
        self.panel = server.gui.add_panel(visible=self.enabled)
        self.panel.float(x=16, y=16, width=420, height=315)
        with self.panel.add_tab("Ego RGB"):
            self.image = server.gui.add_image(
                np.zeros((400, 640, 3), dtype=np.uint8), format="jpeg", jpeg_quality=85
            )
        if self.enabled:
            self.renderer = mujoco.Renderer(model, height=400, width=640)

    def update(self, data: mujoco.MjData) -> None:
        self.panel.visible = self.enabled and self.renderer is not None
        now = time.monotonic()
        if not self.panel.visible or (data.time != 0 and now - self.last_render < 0.1):
            return
        self.renderer.update_scene(data, camera="ego_rgb")
        self.image.image = self.renderer.render()
        self.last_render = now

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
