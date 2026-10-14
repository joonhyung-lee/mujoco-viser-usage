"""Deterministic contact-based pick/place with IK-planned joint trajectories."""

from dataclasses import dataclass

import numpy as np

from .kinematics import SiteIK
from .simulation import Simulation


@dataclass(frozen=True)
class Phase:
    name: str
    joints: np.ndarray
    gripper: float
    duration: float


class PickPlaceTask:
    label = "Pick & place"

    def __init__(self, simulation: Simulation) -> None:
        self.sim = simulation
        self.ik = SiteIK(
            simulation.model,
            "grasp_site",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.phases: list[Phase] = []
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.status = "Ready"
        self.phase_index = 0
        self.sim.data.qfrc_applied[:] = 0

    def start(self) -> None:
        data = self.sim.data
        obj = data.body("pick_object").xpos.copy()
        target = data.site("place_target").xpos.copy()
        target[2] += self.sim.model.geom("pick_object_geom").size[2]
        self.pick_height = float(obj[2])
        qpos = data.qpos.copy()
        orientation = np.diag([1.0, -1.0, -1.0])
        phases = []
        for name, position, opening, duration in [
            ("Approach", obj + [0, 0, 0.22], 1.0, 2.0),
            ("Lower to object", obj + [0, 0, 0.005], 1.0, 2.5),
            ("Close gripper", obj + [0, 0, 0.005], 0.0, 2.0),
            ("Lift", obj + [0, 0, 0.22], 0.0, 2.5),
            ("Transfer", target + [0, 0, 0.22], 0.0, 3.0),
            ("Lower to table", target + [0, 0, 0.005], 0.0, 2.5),
            ("Release", target + [0, 0, 0.005], 1.0, 1.5),
            ("Retract", target + [0, 0, 0.22], 1.0, 2.0),
        ]:
            q = self.ik.solve(qpos, np.asarray(position), orientation)
            qpos[self.ik.qadr] = q
            phases.append(Phase(name, q, opening, duration))
        self.phases = phases
        self.phase_index = 0
        self.active = True
        self.status = phases[0].name
        self._begin_phase()

    def _begin_phase(self) -> None:
        self.started = float(self.sim.data.time)
        self.initial = self.sim.data.ctrl.copy()
        self.status = self.phases[self.phase_index].name

    def update(self) -> None:
        data = self.sim.data
        # Recompute compensation even after completion/manual motor edits.
        data.qfrc_applied[self.ik.dofs] = data.qfrc_bias[self.ik.dofs]
        if not self.active:
            return
        phase = self.phases[self.phase_index]
        fraction = np.clip((data.time - self.started) / phase.duration, 0, 1)
        blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
        goal = np.append(phase.joints, phase.gripper)
        data.ctrl[:] = self.initial + blend * (goal - self.initial)
        if fraction >= 1:
            if (
                phase.name == "Lift"
                and data.body("pick_object").xpos[2] < self.pick_height + 0.1
            ):
                self.active = False
                self.status = "Grasp failed — reset and retry"
                return
            self.phase_index += 1
            if self.phase_index == len(self.phases):
                self.active = False
                error = np.linalg.norm(
                    data.body("pick_object").xpos[:2]
                    - data.site("place_target").xpos[:2]
                )
                expected_z = (
                    data.site("place_target").xpos[2]
                    + self.sim.model.geom("pick_object_geom").size[2]
                )
                on_table = abs(data.body("pick_object").xpos[2] - expected_z) < 0.01
                self.status = (
                    "Complete"
                    if error < 0.04 and on_table
                    else "Placement failed — reset and retry"
                )
            else:
                self._begin_phase()
