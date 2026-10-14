"""Small robotics interface over physics, scene rendering and interactive controls."""

import time
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from .runtime import SimulationRuntime
from .simulation import DEFAULT_SCENE, Simulation
from .tasks import create_task


class ViserParser:
    """Own a model and its optional browser viewer.

    Use run() for an interactive loop, or step()/render() for your own loop.
    These operations, and direct model/data access, belong to one owner thread.
    GUI callbacks are queued and applied by run() or render(). Controls use
    native MuJoCo units (radians for the bundled robot's position actuators).
    """

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_SCENE,
        *,
        viewer: bool = True,
        host: str = "127.0.0.1",
        port: int = 8080,
        pick_place: bool = False,
        navigation: bool = False,
        description: str = "Robot workspace",
        task: str | None = None,
        start_running: bool = True,
        demo: bool = False,
    ) -> None:
        if sum((pick_place, navigation, task is not None)) > 1:
            raise ValueError("Select only one task controller")
        self.simulation = Simulation(Path(xml_path))
        self.runtime = SimulationRuntime(self.simulation)
        self.runtime.running = start_running
        task_name = "pick_place" if pick_place else "navigation" if navigation else task
        self.task = create_task(task_name, self.simulation)
        if self.task is not None:
            self.simulation.before_step = self.task.update
            self.simulation.on_reset = self.task.reset
        if demo:
            if self.task is None:
                raise ValueError("This scene has no scripted demo")
            self.task.start()
        self.server = None
        self.viewer = None
        self.panel = None
        self._closed = False
        if viewer:
            import viser

            from .controls import ControlPanel
            from .viewer import PhysicsViewer

            self.server = viser.ViserServer(host=host, port=port)
            try:
                self.viewer = PhysicsViewer(self.server, self.model)
                self.panel = ControlPanel(
                    self.server, self.runtime, self.viewer, self.task, description
                )
                self.render()
            except Exception:
                self.close()
                raise

    @classmethod
    def from_scene(cls, scene_id: str, **kwargs) -> Self:
        """Open a catalog scene with its matching controller and description."""
        from .scenes import resolve_scene

        scene = resolve_scene(scene_id)
        return cls(
            scene.path,
            task=scene.task,
            description=f"{scene.id} · {scene.description}",
            **kwargs,
        )

    @property
    def model(self) -> mujoco.MjModel:
        return self.simulation.model

    @property
    def data(self) -> mujoco.MjData:
        return self.simulation.data

    def reset(self) -> None:
        self.runtime.reset()
        self.render()

    def set_control(self, actuator: str | int, value: float) -> None:
        index = (
            self.model.actuator(actuator).id if isinstance(actuator, str) else actuator
        )
        self.simulation.set_control(index, value)

    def step(self, ctrl=None, *, nstep: int = 1) -> None:
        if ctrl is not None:
            values = np.asarray(ctrl, dtype=float)
            if values.shape != (self.model.nu,) or not np.isfinite(values).all():
                raise ValueError(f"ctrl must contain {self.model.nu} finite values")
            for index, value in enumerate(values):
                self.simulation.set_control(index, float(value))
        self.simulation.step(nstep)

    def forward(self, qpos=None) -> None:
        """Explicit kinematic pose editing; does not advance simulation time."""
        if qpos is not None:
            values = np.asarray(qpos, dtype=float)
            if values.shape != (self.model.nq,) or not np.isfinite(values).all():
                raise ValueError(f"qpos must contain {self.model.nq} finite values")
            self.data.qpos[:] = values
            mujoco.mj_normalizeQuat(self.model, self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

    def get_q(self) -> np.ndarray:
        return self.data.qpos.copy()

    def get_body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.data.body(name)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    def render(self) -> None:
        """Apply pending UI commands and publish current state without stepping."""
        self.runtime.tick(0.0)
        self._refresh()

    def _refresh(self) -> None:
        if self.viewer is not None:
            self.viewer.update(self.data)
        if self.panel is not None:
            self.panel.refresh()

    def run(self) -> None:
        if self.server is None or self._closed:
            raise RuntimeError("run() requires an open browser viewer")
        previous = time.monotonic()
        try:
            while not self._closed:
                start = time.monotonic()
                if self.runtime.tick(start - previous):
                    self._refresh()
                previous = start
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self.viewer is not None:
                self.viewer.close()
            if self.server is not None:
                self.server.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
