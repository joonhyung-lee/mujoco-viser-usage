"""Four simultaneous worker snapshots in translated display tiles.

Only display coordinates are translated; worker MuJoCo states are never edited.
"""

import mujoco
import numpy as np
import viser

from ..overlays import TrajectoryOverlay
from ..scene_style import configure_scene
from ..viewer import geometry_mesh


class ParallelOverview:
    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel, *, count=4):
        self.server = server
        self.model = model
        self.handles = []
        self.roots = []
        self.labels = []
        self.goals = []
        self.trails = []
        self.count = min(4, count)
        self.site_prefix = (
            ""
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper") >= 0
            else "robot/"
        )
        self.manipulation = (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, self.site_prefix + "left/gripper"
            )
            >= 0
        )
        spacing = (2.0, 1.6) if self.manipulation else (6.0, 2.8)
        self.center = np.array(
            [
                spacing[0] / 2 if self.count > 1 else 0,
                spacing[1] / 2 if self.count > 2 else 0,
                0.1,
            ]
        )
        if not self.manipulation:
            self.center[0] += 2.0
        configure_scene(server)
        server.scene.set_up_direction("+z")
        server.scene.add_grid(
            "/overview/floor",
            width=20,
            height=20,
            plane="xy",
            cell_size=0.5,
            cell_color=(157, 168, 179),
            cell_thickness=0.6,
            section_size=2,
            section_color=(123, 137, 150),
            section_thickness=1.2,
            position=(*self.center[:2], float(model.geom("floor").pos[2])),
            plane_color=(184, 194, 204),
            plane_opacity=1.0,
        )
        meshes = {}
        for i in range(model.ngeom):
            name = model.geom(i).name
            if (
                model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
                or model.geom_group[i] == 3
                or name.startswith("wall_")
                or name in {"start_marker", "goal_marker"}
            ):
                continue
            meshes[i] = geometry_mesh(model, i)
        for worker in range(self.count):
            origin = (spacing[0] * (worker % 2), spacing[1] * (worker // 2), 0.0)
            root = server.scene.add_frame(
                f"/agents/{worker}", position=origin, show_axes=False, visible=False
            )
            self.roots.append(root)
            handles = {}
            for i, mesh in meshes.items():
                material = model.geom_matid[i]
                rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]
                handles[i] = server.scene.add_mesh_simple(
                    f"/agents/{worker}/geom_{i}",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    color=tuple(int(c * 255) for c in rgba[:3]),
                    opacity=float(rgba[3]),
                )
            self.handles.append(handles)
            self.labels.append(
                server.scene.add_label(
                    f"/agents/{worker}/label", f"Worker {worker}", position=(0, 0, 1.0)
                )
            )

            goals, trails = [], []
            if self.manipulation:
                for i, side in enumerate(("left", "right")):
                    goals.append(
                        server.scene.add_icosphere(
                            f"/agents/{worker}/goal_{i}",
                            radius=0.023,
                            color=(118, 185, 0),
                            opacity=0.5,
                        )
                    )
                    trails.append(
                        TrajectoryOverlay(
                            server,
                            model.site(self.site_prefix + side + "/gripper").id,
                            namespace=f"/agents/{worker}/trail_{side}",
                        )
                    )
            self.goals.append(goals)
            self.trails.append(trails)

    def update(self, worker: int, data: mujoco.MjData, label: str, targets=None):
        self.roots[worker].visible = True
        self.labels[worker].text = label
        for trail in self.trails[worker]:
            trail.update(data)
        for i, goal in enumerate(self.goals[worker]):
            goal.visible = targets is not None
            if targets is not None:
                goal.position = targets[i]
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.labels[worker].position = (
            data.xpos[base] + (0, 0, 0.6) if base >= 0 else (0, 0, 0.8)
        )
        with self.server.atomic():
            for i, handle in self.handles[worker].items():
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, data.geom_xmat[i])
                handle.position = data.geom_xpos[i]
                handle.wxyz = quat

    def focus(self, client):
        if client is not None:
            distance = (3.8 if self.count > 1 else 1.8) if self.manipulation else 7.0
            client.camera.position = self.center + np.array(
                [distance * 0.7, -distance, distance * 0.9]
            )
            client.camera.look_at = self.center
            client.camera.fov = np.deg2rad(45)
