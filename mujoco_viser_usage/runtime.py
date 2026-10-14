"""Simulation timing and serialized commands, independent of GUI/network code."""

from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue

import numpy as np

from .simulation import Simulation


@dataclass(frozen=True)
class RuntimeState:
    running: bool
    time: float
    contacts: int
    speed: float
    controls: tuple[float, ...]
    penetration_mm: float


class SimulationRuntime:
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.running = False
        self.speed = 1.0
        self._budget = 0.0
        self._clock_reset = True
        self._commands: SimpleQueue[Callable[[], None]] = SimpleQueue()

    def submit(self, command: Callable[[], None]) -> None:
        """Callbacks enqueue work; tick executes it on the simulation owner thread."""
        self._commands.put(command)

    def toggle(self) -> None:
        self.running = not self.running
        self._budget = 0.0
        self._clock_reset = True

    def reset(self) -> None:
        self.running = False
        self._budget = 0.0
        self._clock_reset = True
        self.simulation.reset()

    def single_step(self) -> None:
        if not self.running:
            self.simulation.step()

    def set_speed(self, speed: float) -> None:
        if not np.isfinite(speed) or not 0.1 <= speed <= 2.0:
            raise ValueError("Speed must be finite and between 0.1 and 2.0")
        self.speed = speed

    def tick(self, elapsed: float) -> bool:
        """Apply pending commands and advance a bounded fixed-step budget."""
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        changed = False
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                break
            command()
            changed = True
        # Do not charge time spent paused before a play/reset command.
        if self._clock_reset:
            elapsed = 0.0
            self._clock_reset = False
        if self.running:
            self._budget += min(elapsed, 0.1) * self.speed
            dt = self.simulation.model.opt.timestep
            steps = int((self._budget + 1e-12) / dt)
            if steps:
                self.simulation.step(steps)
                self._budget = max(0.0, self._budget - steps * dt)
                changed = True
        return changed

    def snapshot(self) -> RuntimeState:
        data = self.simulation.data
        return RuntimeState(
            self.running,
            float(data.time),
            int(data.ncon),
            self.speed,
            tuple(float(x) for x in data.ctrl),
            max(0.0, -float(np.min(data.contact.dist))) * 1000 if data.ncon else 0.0,
        )
