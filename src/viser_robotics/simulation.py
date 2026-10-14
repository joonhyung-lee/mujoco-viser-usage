"""Headless physics. Only the application loop may mutate this state."""

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_SCENE = Path(__file__).parent / "scenes" / "manipulation" / "ur5e" / "scene.xml"
PICK_PLACE_SCENE = DEFAULT_SCENE.with_name("pick_place.xml")


class Simulation:
    def __init__(self, scene: Path = DEFAULT_SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.data = mujoco.MjData(self.model)
        self.before_step: Callable[[], None] | None = None
        self.on_reset: Callable[[], None] | None = None
        self.reset()

    def reset(self) -> None:
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if self.on_reset is not None:
            self.on_reset()
        # The robot's home keyframe predates the free object added by our scene.
        self.drop_box()
        mujoco.mj_forward(self.model, self.data)

    def drop_box(self) -> None:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        if joint >= 0:
            q = self.model.jnt_qposadr[joint]
            v = self.model.jnt_dofadr[joint]
            self.data.qpos[q : q + 7] = self.model.qpos0[q : q + 7]
            self.data.qvel[v : v + 6] = 0
            mujoco.mj_forward(self.model, self.data)

    def set_control(self, index: int, value: float) -> None:
        if not 0 <= index < self.model.nu or not np.isfinite(value):
            raise ValueError("Invalid actuator index or non-finite target")
        if self.model.actuator_ctrllimited[index]:
            value = float(np.clip(value, *self.model.actuator_ctrlrange[index]))
        self.data.ctrl[index] = value

    def step(self, count: int = 1) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Step count must be a positive integer")
        if self.before_step is None:
            mujoco.mj_step(self.model, self.data, nstep=count)
        else:
            for _ in range(count):
                self.before_step()
                mujoco.mj_step(self.model, self.data)
        # mj_step integrates qpos after computing geom transforms. Refresh them
        # so the browser shows the newly integrated state.
        mujoco.mj_forward(self.model, self.data)
