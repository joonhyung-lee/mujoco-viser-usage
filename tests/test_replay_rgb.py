"""Replay RGB thread ownership and stale-frame suppression without requiring GL."""

import threading

import mujoco
import numpy as np
import viser

from viser_robotics.rl.rgb import ReplayCameras


def test_seek_drops_stale_images_and_renderer_stays_on_one_thread(monkeypatch):
    started, release = threading.Event(), threading.Event()
    threads = []

    class Renderer:
        def __init__(self, *_, **__):
            threads.append(threading.get_ident())
            self.first = True

        def update_scene(self, data, camera):
            threads.append(threading.get_ident())
            if self.first:
                self.first = False
                started.set()
                assert release.wait(5)
            self.value = int(data.qpos[0] * 50)

        def render(self):
            threads.append(threading.get_ident())
            return np.full((216, 384, 3), self.value, dtype=np.uint8)

        def close(self):
            threads.append(threading.get_ident())

    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <camera name="robot/wrist_cam_left"/>
      <body><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>""")
    data = [mujoco.MjData(model), mujoco.MjData(model)]
    server = viser.ViserServer(port=0)
    cameras = ReplayCameras(server, model)
    try:
        cameras.update(data)
        assert started.wait(5)
        cameras.invalidate()
        for item in data:
            item.time = 2.0
            item.qpos[0] = 1.0
        release.set()
        cameras.future.result(timeout=5)
        cameras.update(data)
        assert not cameras.image.visible
        cameras.future.result(timeout=5)
        cameras.update(data)
        assert cameras.image.visible
        np.testing.assert_array_equal(
            cameras.image.image, np.full((216, 768, 3), 50, dtype=np.uint8)
        )
        assert cameras.caption.content == "Scripted · 2.00 s | Policy · 2.00 s"
    finally:
        release.set()
        cameras.close()
        cameras.close()
        server.stop()
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
