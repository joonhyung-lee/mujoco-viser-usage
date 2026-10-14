"""Two-arm tabletop pick/place, with contact-only object transport."""

from dataclasses import dataclass

import numpy as np

from ..kinematics import SiteIK
from ..simulation import Simulation


def configure_grasp_contacts(simulation: Simulation) -> None:
    """Firm object/table contacts and a gentle 4 N gripper actuator limit."""
    model = simulation.model
    for i in range(model.ngeom):
        if (
            model.body(model.geom_bodyid[i]).name.startswith("object_")
            or model.geom(i).name == "object_table"
        ):
            model.geom_priority[i] = 2
            model.geom_solref[i] = [0.004, 1.0]
            model.geom_solimp[i, :3] = [0.99, 0.99, 0.001]
    for name in ("left/gripper", "right/gripper"):
        i = model.actuator(name).id
        model.actuator_forcelimited[i] = True
        model.actuator_forcerange[i] = [-4.0, 4.0]


@dataclass
class ArmPhase:
    name: str
    controls: np.ndarray
    seconds: float
    object_id: int
    check_lift: bool = False


class AlohaTask:
    label = "ALOHA pick & place"

    def __init__(self, simulation: Simulation):
        self.sim = simulation
        configure_grasp_contacts(simulation)
        joints = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]
        self.solvers = {
            side: SiteIK(
                simulation.model, side + "/gripper", [side + "/" + j for j in joints]
            )
            for side in ["left", "right"]
        }
        self.targets = np.array(
            [
                simulation.data.site(side + "_target").xpos.copy()
                for side in ("left", "right")
            ]
        )
        self.targets[:, 2] = 0.03
        self.dofs = np.concatenate([ik.dofs for ik in self.solvers.values()])
        self.reset()

    def reset(self):
        self.active = False
        self.status = "Ready · left and right arm pick/place"
        self.phase_index = 0
        self.sim.data.qfrc_applied[:] = 0

    def start(self):
        data = self.sim.data
        qpos = data.qpos.copy()
        controls = data.ctrl.copy()
        phases = []
        self.initial_heights = [data.body(f"object_{i}").xpos[2] for i in range(2)]
        for i, (side, ik) in enumerate(self.solvers.items()):
            obj = data.body(f"object_{i}").xpos.copy()
            goal = self.targets[i]
            sign = 1 if side == "left" else -1
            rotation = np.array([[0, 0, sign], [0, sign, 0], [-1, 0, 0]])
            actuators = np.array(
                [
                    self.sim.model.actuator(side + "/" + j).id
                    for j in [
                        "waist",
                        "shoulder",
                        "elbow",
                        "forearm_roll",
                        "wrist_angle",
                        "wrist_rotate",
                    ]
                ]
            )
            gripper = self.sim.model.actuator(side + "/gripper").id
            for name, pos, opening, duration in [
                ("Approach", obj + [0, 0, 0.16], 0.037, 3.0),
                ("Lower", obj + [0, 0, 0.006], 0.037, 2.5),
                ("Grasp", obj + [0, 0, 0.006], 0.002, 1.5),
                ("Lift", obj + [0, 0, 0.16], 0.002, 2.5),
                ("Transfer", goal + [0, 0, 0.16], 0.002, 3.0),
                ("Place", goal + [0, 0, 0.006], 0.002, 2.5),
                ("Release", goal + [0, 0, 0.006], 0.037, 1.5),
                ("Retract", goal + [0, 0, 0.16], 0.037, 2.0),
            ]:
                q = ik.solve(qpos, np.asarray(pos), rotation)
                qpos[ik.qadr] = q
                controls[actuators] = q
                controls[gripper] = opening
                phases.append(
                    ArmPhase(
                        f"{side.title()}: {name}",
                        controls.copy(),
                        duration,
                        i,
                        name == "Lift",
                    )
                )
        self.phases = phases
        self.phase_index = 0
        self.active = True
        self._begin()

    def _begin(self):
        self.started = self.sim.data.time
        self.initial_controls = self.sim.data.ctrl.copy()
        self.status = self.phases[self.phase_index].name

    def update(self):
        data = self.sim.data
        data.qfrc_applied[self.dofs] = data.qfrc_bias[self.dofs]
        if not self.active:
            return
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
            and data.body(f"object_{phase.object_id}").xpos[2]
            < self.initial_heights[phase.object_id] + 0.08
        ):
            self.active = False
            self.status = f"{phase.name}: grasp failed — reset to retry"
            return
        self.phase_index += 1
        if self.phase_index < len(self.phases):
            self._begin()
        else:
            self.active = False
            errors = [
                np.linalg.norm(data.body(f"object_{i}").xpos - self.targets[i])
                for i in range(2)
            ]
            self.status = (
                "Complete · both objects placed"
                if max(errors) < 0.05
                else "Placement failed — reset to retry"
            )
