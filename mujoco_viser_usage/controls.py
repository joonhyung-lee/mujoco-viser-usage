"""Viser controls translate user intent into queued runtime commands."""

from contextlib import nullcontext
from functools import partial

import mujoco
import numpy as np
import viser

from .runtime import SimulationRuntime
from .tasks import Task
from .viewer import PhysicsViewer


def angle_actuator(model: mujoco.MjModel, index: int) -> bool:
    """Degrees are valid only for a unit-gear hinge position actuator."""
    if model.actuator_trntype[index] != mujoco.mjtTrn.mjTRN_JOINT:
        return False
    joint = model.actuator_trnid[index, 0]
    gain = model.actuator_gainprm[index, 0]
    return bool(
        model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
        and model.actuator_gaintype[index] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[index] == mujoco.mjtBias.mjBIAS_AFFINE
        and model.actuator_dyntype[index] == mujoco.mjtDyn.mjDYN_NONE
        and np.allclose(model.actuator_gear[index], [1, 0, 0, 0, 0, 0])
        and gain > 0
        and np.isclose(model.actuator_biasprm[index, 1], -gain)
    )


def actuator_group(name: str) -> str:
    """Readable joint groups for bilateral arms and quadruped legs."""
    prefix = name.lower().replace("/", "_").split("_")[0]
    return {
        "left": "Left arm",
        "right": "Right arm",
        "arm": "Arm & gripper",
        "fl": "Front left leg",
        "fr": "Front right leg",
        "hl": "Rear left leg",
        "rl": "Rear left leg",
        "hr": "Rear right leg",
        "rr": "Rear right leg",
    }.get(prefix, "Other joints")


class ControlPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        runtime: SimulationRuntime,
        viewer: PhysicsViewer,
        task: Task | None = None,
        description: str = "Robot workspace",
    ) -> None:
        self.task = task
        self.runtime = runtime
        self.viewer = viewer
        self.model = runtime.simulation.model
        gui = server.gui
        gui.add_markdown(f"### Robot workspace\n\n{description}")
        self.status = gui.add_markdown("")
        self.play = gui.add_button("Start simulation", icon=viser.Icon.PLAYER_PLAY)
        self.play.on_click(lambda _: runtime.submit(runtime.toggle))
        self.actions = gui.add_button_group("", options=("Step", "Reset"))
        self.actions.on_click(
            lambda event: runtime.submit(
                runtime.single_step if event.target.value == "Step" else runtime.reset
            )
        )
        self.speed = gui.add_dropdown(
            "Speed", options=("0.25×", "0.5×", "1×", "2×"), initial_value="1×"
        )
        self.speed.on_update(
            lambda event: runtime.submit(
                partial(runtime.set_speed, float(event.target.value[:-1]))
            )
        )
        self.task_button = None
        self.task_status = None
        if task is not None:
            self.task_button = gui.add_button(
                f"Run {task.label.lower()}", icon=viser.Icon.ROBOT
            )
            self.task_button.on_click(lambda _: runtime.submit(self._start_task))
            self.task_status = gui.add_markdown("")
        self.sliders = []
        with gui.add_folder("Joints", expand_by_default=False):
            self.hint = gui.add_markdown("")
            folders = {}
            for i in range(self.model.nu):
                group = (
                    actuator_group(self.model.actuator(i).name)
                    if self.model.nu > 10
                    else None
                )
                if group is not None and group not in folders:
                    folders[group] = gui.add_folder(
                        group, expand_by_default=not folders
                    )
                with folders[group] if group is not None else nullcontext():
                    factor = 180 / np.pi if angle_actuator(self.model, i) else 1.0
                    unit = "°" if factor != 1 else "ctrl"
                    if task is not None and self.model.actuator(i).name == "gripper":
                        factor, unit = 100.0, "% open"
                    name = (
                        (self.model.actuator(i).name or f"Motor {i + 1}")
                        .replace("_", " ")
                        .title()
                    )
                    initial = float(runtime.simulation.data.ctrl[i]) * factor
                    if self.model.actuator_ctrllimited[i]:
                        low, high = self.model.actuator_ctrlrange[i] * factor
                        if factor != 1:
                            # Friendly degree ticks; Simulation enforces exact limits.
                            low, high = np.floor(low * 10) / 10, np.ceil(high * 10) / 10
                        slider = gui.add_slider(
                            f"{name} ({unit})",
                            min=float(low),
                            max=float(high),
                            step=1.0 if factor != 1 else 0.01,
                            initial_value=initial,
                        )
                    else:
                        # Unbounded motors need an input, not invented slider limits.
                        slider = gui.add_number(
                            f"{name} ({unit})", initial_value=initial, step=0.01
                        )
                    slider.precision = 1 if factor != 1 else 3
                    slider.on_update(
                        partial(self._control_changed, index=i, factor=factor)
                    )
                    self.sliders.append((slider, factor))
            if not self.sliders:
                gui.add_markdown("This model has no actuators.")
        with gui.add_folder("View", expand_by_default=False):
            camera = gui.add_button("Reset camera", icon=viser.Icon.HOME)
            camera.on_click(lambda event: viewer.reset_camera(event.client))
            collisions = gui.add_checkbox("Collision shapes", initial_value=False)
            collisions.on_update(
                lambda event: runtime.submit(
                    partial(viewer.show_collisions, event.target.value)
                )
            )
            for label, initial, callback in (
                ("Contact forces", True, viewer.show_forces),
                ("Tracked trajectory · 10 anchors", True, viewer.show_trajectory),
                ("Ego RGB overlay", viewer.ego.enabled, viewer.show_ego),
            ):
                toggle = gui.add_checkbox(label, initial_value=initial)
                toggle.on_update(
                    lambda event, fn=callback: runtime.submit(
                        partial(fn, event.target.value)
                    )
                )
            gui.add_markdown(
                "**Force arrows:** 1 N = 2 cm, capped at 25 cm.\n\n**Trajectory:** measured tool / base path; 10 equally spaced anchors."
            )
            contacts = gui.add_checkbox("Contact points", initial_value=False)
            contacts.on_update(
                lambda event: runtime.submit(
                    partial(viewer.show_contacts, event.target.value)
                )
            )
            frame = gui.add_checkbox(
                "Tool coordinate frame",
                initial_value=viewer.site_id >= 0,
                disabled=viewer.site_id < 0,
            )
            frame.on_update(
                lambda event: runtime.submit(
                    partial(viewer.show_tool_frame, event.target.value)
                )
            )
            if (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
                >= 0
            ):
                drop = gui.add_button("Drop test box")
                drop.on_click(lambda _: runtime.submit(runtime.simulation.drop_box))
            gui.add_markdown("Metric grid · Drag to orbit · Scroll to zoom")
        with gui.add_folder("State", expand_by_default=False):
            self.telemetry = gui.add_markdown("")
        self._last_controls = None
        self.refresh()

    def _start_task(self) -> None:
        if self.task is None or self.task.active:
            return
        try:
            self.task.start()
        except ValueError as error:
            self.task.status = f"Cannot plan: {error}"
            return
        if not self.runtime.running:
            self.runtime.toggle()

    def _control_changed(self, event, *, index: int, factor: float) -> None:
        # Programmatic reset synchronization must not enqueue stale controls.
        if event.client is not None and not (self.task and self.task.active):
            value = float(event.target.value) / factor
            self.runtime.submit(
                partial(self.runtime.simulation.set_control, index, value)
            )

    def refresh(self) -> None:
        state = self.runtime.snapshot()
        if self.task is not None:
            self.task_status.content = f"**{self.task.label}:** {self.task.status}"
            self.task_button.disabled = self.task.active
            for slider, _ in self.sliders:
                slider.disabled = self.task.active
        self.status.content = (
            f"**{'● Running' if state.running else 'Ⅱ Paused'}** · "
            f"{state.time:.3f} s · {state.contacts} contacts"
        )
        self.play.label = "Pause simulation" if state.running else "Start simulation"
        self.play.icon = (
            viser.Icon.PLAYER_PAUSE if state.running else viser.Icon.PLAYER_PLAY
        )
        self.hint.content = (
            "Adjust motor targets. Pause before using Step."
            if state.running
            else (
                "Set targets, then press **Start simulation**. "
                f"**Step** advances {self.model.opt.timestep * 1000:g} ms."
            )
        )
        if state.controls != self._last_controls:
            for (slider, factor), value in zip(self.sliders, state.controls):
                slider.value = value * factor
            self._last_controls = state.controls
        rows = ["| Joint | Actual |", "| :-- | --: |"]
        data = self.runtime.simulation.data
        for i in range(self.model.njnt):
            kind = self.model.jnt_type[i]
            if kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            value = data.qpos[self.model.jnt_qposadr[i]]
            unit = "m"
            if kind == mujoco.mjtJoint.mjJNT_HINGE:
                value, unit = np.rad2deg(value), "°"
            name = (
                (self.model.joint(i).name or str(i))
                .replace("_joint", "")
                .replace("_", " ")
            )
            rows.append(f"| {name} | {value:.2f} {unit} |")
        self.telemetry.content = (
            "\n".join(rows) + f"\n\nPhysics step: {self.model.opt.timestep * 1000:g} ms"
            f"\n\nMax contact penetration: **{state.penetration_mm:.3f} mm**"
            "\n\nDynamics: `mj_step` · IK computes motor targets only."
        )
