"""Closed-loop wheel velocity control. Motion comes exclusively from mj_step."""

import mujoco
import numpy as np

from ...simulation import Simulation
from ..maze import route


class MazeNavigationTask:
    label = "Navigate maze"

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.path = route()
        # Derive kinematic dimensions from this model, not a robot-name switch.
        model = simulation.model
        wheel_name = (
            "front_left_wheel_link"
            if mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "front_left_wheel_link"
            )
            >= 0
            else "left_0"
        )
        wheel = model.body(wheel_name)
        self.halftrack = abs(float(wheel.pos[1]))
        collision = next(
            i
            for i in range(
                int(wheel.geomadr[0]), int(wheel.geomadr[0] + wheel.geomnum[0])
            )
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        self.radius = float(model.geom_size[collision, 0])
        self.left = np.array(
            [
                "left" in simulation.model.actuator(i).name
                for i in range(simulation.model.nu)
            ]
        )
        self.reset()

    def reset(self):
        self.active = False
        self.status = "Ready · shortest grid route, wheel velocity control"
        self.index = 1

    def start(self):
        self.index = min(
            int(
                np.argmin(
                    np.linalg.norm(
                        self.path - self.simulation.data.body("base").xpos[:2], axis=1
                    )
                )
            )
            + 1,
            len(self.path) - 1,
        )
        self.started_at = self.simulation.data.time
        self.active = True
        self.status = "Following route"

    def update(self):
        if not self.active:
            return
        data = self.simulation.data
        body = data.body("base")
        if data.time - self.started_at > 180 or body.xmat[8] < 0.5:
            data.ctrl[:] = 0
            self.active = False
            self.status = "Stopped: timeout or robot tipped; reset to retry"
            return
        delta = self.path[self.index] - body.xpos[:2]
        distance = float(np.linalg.norm(delta))
        if distance < 0.13:
            if self.index == len(self.path) - 1:
                data.ctrl[:] = 0
                self.active = False
                self.status = "Goal reached"
                return
            self.index += 1
            delta = self.path[self.index] - body.xpos[:2]
            distance = float(np.linalg.norm(delta))
        rotation = body.xmat.reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        error = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        speed = (
            min(0.38, 0.9 * distance) * max(0.0, np.cos(error))
            if abs(error) < 0.5
            else 0.0
        )
        turn = float(np.clip(2.5 * error, -1.2, 1.2))
        targets = (
            speed + np.where(self.left, -1.0, 1.0) * self.halftrack * turn
        ) / self.radius
        data.ctrl[:] = np.clip(
            targets,
            self.simulation.model.actuator_ctrlrange[:, 0],
            self.simulation.model.actuator_ctrlrange[:, 1],
        )
        self.status = f"Waypoint {self.index}/{len(self.path) - 1} · {distance:.2f} m"
