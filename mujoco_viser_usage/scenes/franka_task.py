"""Franka contact-rich tasks scripted through IK: plug insertion, cable routing, NIST board."""

from dataclasses import dataclass

import numpy as np

from ..kinematics import SiteIK
from ..simulation import Simulation

JOINTS = [f"joint{i}" for i in range(1, 8)]
OPEN, CLOSED = 255.0, 0.0
LABELS = {
    "plug": "Franka plug insertion",
    "cable": "Franka cable routing",
    "nist": "NIST task board assembly",
}
# NIST subset for the scripted run: part, finger axis (-y: fingers clear the 40 mm
# tray pitch and joint 7 stays off its limit), grasp/seat TCP offset above
# the part center. Pads span TCP -0.009 .. +0.008 m, so offsets keep tips clear.
NIST_SEQUENCE = [
    ("rod_16", (0, -1, 0), 0.012),
    ("peg_16", (0, -1, 0), 0.012),
    ("rod_12", (0, -1, 0), 0.012),
    ("gear_large", (0, -1, 0), 0.010),  # origin at the bottom face
    ("nut_16", (0, -1, 0), 0.009),
]


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


def pick_place(name, grasp, seat, rotation, lift=0.12, part=None, carry=0.5):
    """Approach / lower / grasp / lift / carry / seat / release / retract."""
    up = np.array([0, 0, lift])
    return [
        (f"{name}: approach", grasp + up, rotation, OPEN, 2.5, None),
        (f"{name}: lower", grasp, rotation, OPEN, 2.0, None),
        (f"{name}: grasp", grasp, rotation, CLOSED, 1.0, None),
        (f"{name}: lift", grasp + up, rotation, CLOSED, 2.0, part),
        (f"{name}: carry", seat + up * carry, rotation, CLOSED, 3.0, None),
        (f"{name}: seat", seat, rotation, CLOSED, 3.0, None),
        (f"{name}: release", seat, rotation, OPEN, 1.0, None),
        (f"{name}: retract", seat + up, rotation, OPEN, 2.0, None),
    ]


class FrankaTask:
    def __init__(self, simulation: Simulation, variant: str):
        self.sim = simulation
        self.variant = variant
        self.label = LABELS[variant]
        model = simulation.model
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
            seat = socket + [0, 0, handle[2] - data.site("plug_tip").xpos[2] + 0.001]
            return pick_place(
                "Plug", handle, seat, grasp_rotation([1, 0, 0]), part="plug"
            )
        if self.variant == "cable":
            end = data.body("cable_end").xpos + [0, 0, 0.004]
            clip = data.site("clip_center").xpos.copy()
            return pick_place(
                "Cable",
                end,
                clip,
                grasp_rotation([0, 1, 0]),
                lift=0.08,
                part="cable_end",
                carry=1.0,
            )
        steps = []
        for part, axis, offset in NIST_SEQUENCE:
            grasp = data.body(part).xpos + [0, 0, offset]
            seat = data.site(f"{part}_goal").xpos + [0, 0, offset]
            steps += pick_place(part, grasp, seat, grasp_rotation(axis), part=part)
        return steps

    def start(self):
        data = self.sim.data
        qpos = data.qpos.copy()
        controls = data.ctrl.copy()
        self.initial_heights = {}
        self.phases = []
        for name, pos, rotation, opening, seconds, check in self._plan():
            try:
                q = self.ik.solve(qpos, np.asarray(pos, dtype=float), rotation)
            except ValueError:
                # Seeding from the previous pose can trap the damped solver near a
                # joint limit; the home keyframe reaches the whole table top-down.
                home = qpos.copy()
                home[self.ik.qadr] = self.sim.model.key_qpos[0][self.ik.qadr]
                q = self.ik.solve(home, np.asarray(pos, dtype=float), rotation)
            qpos[self.ik.qadr] = q
            controls[self.actuators] = q
            controls[self.gripper] = opening
            self.phases.append(Phase(name, controls.copy(), seconds, check))
            if check:
                self.initial_heights[check] = data.body(check).xpos[2]
        self.phase_index = 0
        self.active = True
        self._begin()

    def _begin(self):
        self.started = self.sim.data.time
        self.initial_controls = self.sim.data.ctrl.copy()
        self.status = self.phases[self.phase_index].name

    def seated(self):
        """Parts within 1 mm of their assembled pose (AlphaNIST criterion)."""
        data = self.sim.data
        if self.variant == "plug":
            error = data.site("plug_tip").xpos - data.site("socket_bottom").xpos
            ok = np.linalg.norm(error[:2]) < 0.004 and error[2] < 0.004
            return ["plug"] if ok else []
        if self.variant == "cable":
            # Inside the U-channel: between the walls (y), along it (x), on the table (z).
            error = np.abs(data.body("cable_end").xpos - data.site("clip_center").xpos)
            return ["cable_end"] if (error < [0.03, 0.01, 0.01]).all() else []
        return [
            part
            for part, _, _ in NIST_SEQUENCE
            if np.linalg.norm(data.body(part).xpos - data.site(f"{part}_goal").xpos)
            < 0.001
        ]

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
            return
        self.active = False
        done = self.seated()
        total = len(NIST_SEQUENCE) if self.variant == "nist" else 1
        self.status = (
            f"Complete · {len(done)}/{total} seated"
            if len(done) == total
            else f"{len(done)}/{total} seated — reset to retry"
        )
