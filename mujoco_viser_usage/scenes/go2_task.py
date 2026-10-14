"""Analytic trot sample; joint servos and ground contacts produce all motion."""

import numpy as np

from ..simulation import Simulation


def gait_targets(
    time: float,
    stride: float = 0.2,
    frequency: float = 2.0,
    lift: float = 0.08,
    height: float = 0.28,
) -> np.ndarray:
    """Planar two-link leg IK, diagonal pairs offset by half a gait cycle."""
    result = []
    for offset in (0.0, 0.5, 0.5, 0.0):
        phase = (time * frequency + offset) % 1.0
        if phase < 0.6:
            x, z = stride * (0.5 - phase / 0.6), -height
        else:
            swing = (phase - 0.6) / 0.4
            x, z = stride * (-0.5 + swing), -height + lift * np.sin(np.pi * swing)
        knee = -np.arccos(
            np.clip((x * x + z * z - 2 * 0.213**2) / (2 * 0.213**2), -1.0, 1.0)
        )
        thigh = np.arctan2(-x, -z) - knee / 2
        result.extend((0.0, thigh, knee))
    return np.asarray(result)


def configure_servos(simulation: Simulation):
    """Keep the model's torque bounds while using gains suited to this trot."""
    simulation.model.actuator_gainprm[:, 0] = 150.0
    simulation.model.actuator_biasprm[:, 1] = -150.0
    simulation.model.actuator_biasprm[:, 2] = -4.0


class Go2WalkTask:
    label = "Go2 walk sample"

    def __init__(self, simulation: Simulation):
        self.sim = simulation
        configure_servos(simulation)
        self.home = simulation.data.ctrl.copy()
        self.reset()

    def reset(self):
        self.active = False
        self.status = "Ready · contact-driven trot along the first maze corridor"
        self.braking_at = None

    def start(self):
        self.started = self.sim.data.time
        self.initial = self.sim.data.ctrl.copy()
        self.origin = self.sim.data.body("base").xpos.copy()
        self.braking_at = None
        self.active = True
        self.status = "Walking · target 2 m forward"

    def update(self):
        if not self.active:
            return
        data = self.sim.data
        elapsed = data.time - self.started
        body = data.body("base")
        progress = body.xpos[0] - self.origin[0]
        if body.xpos[2] < 0.16 or body.xmat[8] < 0.65 or elapsed > 20:
            self.active = False
            data.ctrl[:] = self.home
            self.status = "Stopped · fall or timeout; reset to retry"
            return
        if progress >= 2.0 and self.braking_at is None:
            self.braking_at = data.time
            self.braking_controls = data.ctrl.copy()
        if self.braking_at is not None:
            blend = np.clip((data.time - self.braking_at) / 1.0, 0, 1)
            data.ctrl[:] = self.braking_controls * (1 - blend) + self.home * blend
            self.status = f"Settling · {progress:.2f} m"
            if data.time - self.braking_at >= 2:
                self.active = False
                self.status = f"Complete · walked {progress:.2f} m"
            return
        ramp = np.clip(elapsed, 0, 1)
        data.ctrl[:] = self.initial * (1 - ramp) + gait_targets(elapsed) * ramp
        self.status = f"Walking · {progress:.2f} / 2.00 m"
