"""Record clean GIFs: in-process physics, a bare Viser scene, headed Chromium frames.

Usage:
  uv run python -m mujoco_viser_usage.tools.record_gif nist
  uv run --extra rl python -m mujoco_viser_usage.tools.record_gif insert --run runs/<train-run>

No GUI panel, no toasts; one screenshot per simulated frame, so playback speed
is exact regardless of browser frame rate.
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import viser
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from ..scenes import resolve_scene
from ..simulation import Simulation
from ..viewer import PhysicsViewer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/assets"
# Hide every overlay (panel, minimized tab, toasts): anything positioned that
# is not the canvas or one of its ancestors.
HIDE_UI = """
for (const e of document.querySelectorAll('body *')) {
  const pos = getComputedStyle(e).position;
  if (e.tagName !== 'CANVAS' && !e.querySelector('canvas') && (pos === 'fixed' || pos === 'absolute'))
    e.style.display = 'none';
}
"""


class Recorder:
    def __init__(self, model, camera, size, port=8111):
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = PhysicsViewer(self.server, model, connect_camera=False)
        for show in (
            self.viewer.show_tool_frame,
            self.viewer.show_contacts,
            self.viewer.show_forces,
            self.viewer.show_trajectory,
        ):
            show(False)
        self.server.gui.main_panel.minimize()
        position, look_at = camera

        @self.server.on_client_connect
        def _(client):
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.fov = np.deg2rad(40)

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page(
            viewport={"width": size[0], "height": size[1]}
        )
        for _ in range(40):
            try:
                self.page.goto(f"http://127.0.0.1:{port}", timeout=2000)
                break
            except PlaywrightError:
                time.sleep(0.5)
        self.page.wait_for_timeout(2500)
        self.page.evaluate(HIDE_UI)
        self.page.wait_for_timeout(300)
        self.frames = []

    def frame(self, data):
        self.viewer.update(data)
        self.page.wait_for_timeout(60)  # let the client apply the poses
        self.frames.append(
            Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        )

    def save(self, path, fps, width, hold=1.0):
        frames = [
            f.resize((width, round(width * f.height / f.width)), Image.LANCZOS)
            for f in self.frames
        ]
        sample = frames[:: max(1, len(frames) // 6)]
        strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for i, f in enumerate(sample):
            strip.paste(f, (0, i * f.height))
        palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        quantized = [
            f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
            for f in frames
        ]
        durations = [round(1000 / fps)] * len(quantized)
        durations[-1] += round(1000 * hold)
        quantized[0].save(
            path,
            save_all=True,
            append_images=quantized[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
        print(
            path, path.stat().st_size // 1000, "KB", len(frames), "frames", flush=True
        )

    def close(self):
        self.browser.close()
        self.playwright.stop()
        self.server.stop()


def record_nist(args):
    from ..scenes.franka_task import FrankaTask

    sim = Simulation(resolve_scene("manipulation/nist_board").path)
    task = FrankaTask(sim, "nist")
    recorder = Recorder(
        sim.model, ((1.25, -0.75, 0.75), (0.5, -0.05, 0.05)), (1600, 900)
    )
    try:
        task.start()
        per_frame = round(args.speed / args.fps / sim.model.opt.timestep)
        while task.active and sim.data.time < 120:
            for _ in range(per_frame):
                task.update()
                sim.step()
            recorder.frame(sim.data)
        recorder.save(OUT / "nist-board.gif", args.fps, args.width)
    finally:
        recorder.close()


def record_insert(args):
    from stable_baselines3 import PPO

    from ..rl.config import TrainConfig
    from ..rl.envs import make_env

    run = Path(args.run)
    env = make_env(TrainConfig.load(run / "config.json"))
    policy = PPO.load(run / "checkpoints/final.zip", device="cpu")
    recorder = Recorder(
        env.model, ((0.86, -0.24, 0.17), (0.594, -0.07, 0.03)), (1600, 900)
    )
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=1000003 + episode)
            recorder.frame(env.data)
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                recorder.frame(env.data)
                if terminated or truncated:
                    break
            for _ in range(round(0.6 / env.config.control_dt)):  # hold the seated part
                recorder.frame(env.data)
            print("episode", episode, "success", info["is_success"], flush=True)
        recorder.save(
            OUT / "franka-insert.gif", round(1 / env.config.control_dt), args.width
        )
    finally:
        recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["nist", "insert"])
    parser.add_argument("--run")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--speed", type=float, default=6.0, help="nist: sim seconds per GIF second"
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1000)
    args = parser.parse_args()
    (record_nist if args.mode == "nist" else record_insert)(args)


if __name__ == "__main__":
    main()
