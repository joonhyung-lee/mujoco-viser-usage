"""Franka contact-rich tasks: plug insertion and cable-end routing, scripted via IK."""

from dataclasses import dataclass

import numpy as np

from ..kinematics import SiteIK
from ..simulation import Simulation

JOINTS = [f"joint{i}" for i in range(1, 8)]
OPEN, CLOSED = 255.0, 0.0


@dataclass
class Phase:
    name: str
    controls: np.ndarray
    seconds: float
    check_lift: str | None = None


def grasp_rotation(finger_axis):
    """TCP frame with z pointing down and the fingers closing along finger_axis."""
    y = np.asarray(finger_axis, dtype=float)
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack((np.cross(y, z), y, z))


class FrankaTask:
    def __init__(self, simulation: Simulation, variant: str):
        self.sim = simulation
        self.variant = variant
        model = simulation.model
        self.label = (
            "Franka plug insertion" if variant == "plug" else "Franka cable routing"
        )
        self.ik = SiteIK(model, "tcp", JOINTS)
        self.actuators = np.array(
            [model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self.gripper = model.actuator("actuator8").id
        # Upstream gripper force is proportional to opening (100 N/m), so a 24 mm
        # part gets 2.4 N. Stiffen 4x, keep the 40 mm open width and a 25 N cap.
        model.actuator_gainprm[self.gripper, 0] *= 4
        model.actuator_biasprm[self.gripper, 1:3] *= 4
        model.actuator_forcerange[self.gripper] = [-25.0, 25.0]
        self.reset()

    def reset(self):
        self.active = False
        self.status = f"Ready · {self.label}"
        self.phase_index = 0

    def _plan(self):
        data = self.sim.data
        if self.variant == "plug":
            handle = data.body("plug").xpos.copy()
            socket = data.site("socket_bottom").xpos.copy()
            tip_offset = handle[2] - data.site("plug_tip").xpos[2]
            seat = socket + [0, 0, tip_offset + 0.001]
            rotation = grasp_rotation([1, 0, 0])
            return rotation, [
                ("Approach", handle + [0, 0, 0.12], OPEN, 2.5, None),
                ("Lower", handle, OPEN, 2.0, None),
                ("Grasp", handle, CLOSED, 1.0, None),
                ("Lift", handle + [0, 0, 0.12], CLOSED, 2.0, "plug"),
                ("Align", seat + [0, 0, 0.06], CLOSED, 3.0, None),
                ("Insert", seat, CLOSED, 3.0, None),
                ("Release", seat, OPEN, 1.0, None),
                ("Retract", seat + [0, 0, 0.12], OPEN, 2.0, None),
            ]
        end = data.body("cable_end").xpos.copy()
        clip = data.site("clip_center").xpos.copy()
        rotation = grasp_rotation([0, 1, 0])
        return rotation, [
            ("Approach", end + [0, 0, 0.12], OPEN, 2.5, None),
            ("Lower", end + [0, 0, 0.004], OPEN, 2.0, None),
            ("Grasp", end + [0, 0, 0.004], CLOSED, 1.0, None),
            ("Lift", end + [0, 0, 0.08], CLOSED, 2.0, "cable_end"),
            ("Route", clip + [0, 0, 0.08], CLOSED, 4.0, None),
            ("Seat", clip, CLOSED, 2.5, None),
            ("Release", clip, OPEN, 1.0, None),
            ("Retract", clip + [0, 0, 0.12], OPEN, 2.0, None),
        ]

    def start(self):
        data = self.sim.data
        qpos = data.qpos.copy()
        controls = data.ctrl.copy()
        rotation, steps = self._plan()
        self.initial_heights = {
            name: data.body(name).xpos[2]
            for name in ("plug", "cable_end")
            if name
            in [self.sim.model.body(i).name for i in range(self.sim.model.nbody)]
        }
        self.phases = []
        for name, pos, opening, seconds, check in steps:
            q = self.ik.solve(qpos, np.asarray(pos), rotation)
            qpos[self.ik.qadr] = q
            controls[self.actuators] = q
            controls[self.gripper] = opening
            self.phases.append(Phase(name, controls.copy(), seconds, check))
        self.phase_index = 0
        self.active = True
        self._begin()

    def _begin(self):
        self.started = self.sim.data.time
        self.initial_controls = self.sim.data.ctrl.copy()
        self.status = self.phases[self.phase_index].name

    def _success(self):
        data = self.sim.data
        if self.variant == "plug":
            error = data.site("plug_tip").xpos - data.site("socket_bottom").xpos
            return np.linalg.norm(error[:2]) < 0.004 and error[2] < 0.004
        # Inside the U-channel: between the walls (y), along it (x), on the table (z).
        error = np.abs(data.body("cable_end").xpos - data.site("clip_center").xpos)
        return bool((error < [0.03, 0.01, 0.01]).all())

    def update(self):
        if not self.active:
            return
        data = self.sim.data
        phase = self.phases[self.phase_index]
        fraction = np.clip((data.time - self.started) / phase.seconds, 0, 1)
        blend = fraction**3 * (10 - 15 * fraction + 6 * fraction * fraction)
        data.ctrl[:] = self.initial_controls + blend * (
            phase.controls - self.initial_controls
        )
        if fraction < 1:
            return
        if (
            phase.check_lift
            and data.body(phase.check_lift).xpos[2]
            < self.initial_heights[phase.check_lift] + 0.05
        ):
            self.active = False
            self.status = f"{phase.name}: grasp failed — reset to retry"
            return
        self.phase_index += 1
        if self.phase_index < len(self.phases):
            self._begin()
        else:
            self.active = False
            self.status = (
                "Complete · "
                + ("plug seated" if self.variant == "plug" else "cable clipped")
                if self._success()
                else "Not seated — reset to retry"
            )
