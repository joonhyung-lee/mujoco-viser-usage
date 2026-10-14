"""Paired replay cameras, with one thread owning the OpenGL renderer."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

import mujoco
import numpy as np


class ReplayCameras:
    """Render recorded states; stale frames from a seek are never published."""

    def __init__(self, server, model):
        self._lock = RLock()
        self.closed = False
        self.server, self.model = server, model
        candidates = {
            "Left wrist": "robot/wrist_cam_left",
            "Right wrist": "robot/wrist_cam_right",
            "Overview": "robot/teleoperator_pov",
        }
        self.cameras = {
            label: name
            for label, name in candidates.items()
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
        }
        self.available = bool(self.cameras)
        self.enabled = self.available
        self.renderer = None
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-rgb")
            if self.available
            else None
        )
        self.error = None
        self.future = None
        self.version = 0
        self.last_request = -float("inf")
        self.last_key = None
        self.panel = server.gui.add_panel(visible=self.available)
        self.panel.float(x=16, y=16, width=600, height=300)
        with self.panel.add_tab("Replay RGB"):
            self.camera = server.gui.add_dropdown(
                "Camera", options=tuple(self.cameras) or ("Unavailable",)
            )
            self.image = server.gui.add_image(
                np.zeros((216, 768, 3), dtype=np.uint8), format="jpeg", jpeg_quality=85
            )
            self.caption = server.gui.add_markdown("Scripted · left | Policy · right")
        self.camera.on_update(lambda _: self.invalidate())

    def invalidate(self):
        with self._lock:
            if self.closed:
                return
            self.error = None
            self.version += 1
            self.last_key = None
            self.last_request = -float("inf")
            # Do not leave a previous timestamp visible while a seek is rendering.
            self.caption.content = "Updating replay cameras…"
            self.image.visible = False

    def update(self, data):
        with self._lock:
            if self.closed:
                return
            self.panel.visible = self.enabled and self.available
            if not self.panel.visible or self.error is not None:
                return
            if self.future is not None:
                if not self.future.done():
                    return
                try:
                    version, pixels, times = self.future.result()
                except Exception as error:
                    logging.getLogger(__name__).exception("Replay RGB rendering failed")
                    self.future = None
                    self.error = str(error)
                    self.image.visible = False
                    self.caption.content = f"RGB unavailable: {error}"
                    return
                self.future = None
                if version == self.version:
                    self.image.image = pixels
                    self.image.visible = True
                    self.caption.content = (
                        f"Scripted · {times[0]:.2f} s | Policy · {times[1]:.2f} s"
                    )
            key = (
                self.version,
                self.camera.value,
                *(float(item.time) for item in data),
            )
            if key == self.last_key or time.monotonic() - self.last_request < 0.1:
                return
            snapshots = [
                {
                    name: getattr(item, name).copy()
                    for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")
                }
                for item in data
            ]
            times = tuple(float(item.time) for item in data)
            self.future = self.executor.submit(
                self._render,
                self.version,
                self.cameras[self.camera.value],
                snapshots,
                times,
            )
            self.last_key = key
            self.last_request = time.monotonic()

    def _render(self, version, camera, snapshots, times):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=216, width=384)
        frames = []
        for snapshot in snapshots:
            data = mujoco.MjData(self.model)
            for name, value in snapshot.items():
                getattr(data, name)[:] = value
            mujoco.mj_forward(self.model, data)
            self.renderer.update_scene(data, camera=camera)
            frames.append(self.renderer.render().copy())
        return version, np.concatenate(frames, axis=1), times

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self.executor is not None:
                # Renderer creation, use, and destruction must share an OS thread.
                def release():
                    if self.renderer is not None:
                        self.renderer.close()
                        self.renderer = None

                self.executor.submit(release).result()
                self.executor.shutdown(wait=True)
                self.executor = None
            self.panel.remove()
