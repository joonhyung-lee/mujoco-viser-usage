"""Capture README/site screenshots from live servers with a headed Chromium.

Headed Chromium uses hardware WebGL, so Viser shows no software-rendering toast.
Requires recorded runs under runs for the dashboard and comparison shots.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/assets"
APP = [sys.executable, "-m", "mujoco_viser_usage.app"]
DASHBOARD = [sys.executable, "-m", "mujoco_viser_usage.rl.dashboard"]


def select_scene(page, label):
    """Pick a dashboard scene in the Viser dropdown, then let rollouts buffer."""
    page.locator('input[value*="·"]').first.click()
    page.get_by_text(label, exact=True).click()
    page.wait_for_timeout(6000)


SHOTS = [
    ("pick-place", APP + ["manipulation/ur5e", "--demo"], 13, None),
    ("aloha", APP + ["manipulation/aloha", "--demo"], 22, None),
    ("husky-maze", APP + ["navigation/husky_mixed", "--demo"], 45, None),
    ("rl-dashboard", DASHBOARD, 8, None),
    ("rl-go2", DASHBOARD, 4, lambda page: select_scene(page, "Go2 · walk")),
]


def latest(pattern):
    return max(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)


def capture(page, name, command, wait, port, action=None):
    process = subprocess.Popen(
        [*command, "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                page.goto(f"http://127.0.0.1:{port}", timeout=2000)
                break
            except PlaywrightError:
                time.sleep(0.5)
        page.wait_for_timeout(1000 * wait)
        if action:
            action(page)
        toast = page.get_by_text("Software WebGL rendering detected")
        if toast.count():
            toast.locator("xpath=../..").get_by_role("button").first.click()
            page.wait_for_timeout(300)
        path = OUT / f"{name}.png"
        page.screenshot(path=path)
        image = Image.open(path)
        image.resize((1200, round(1200 * image.height / image.width))).save(path)
        print(name, path.stat().st_size // 1000, "KB", flush=True)
    finally:
        process.terminate()
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shots = list(SHOTS)
    compare = [sys.executable, "-m", "mujoco_viser_usage.rl.comparison"]
    for name, pattern in [
        ("mjlab-comparison", "runs/*mjlab-compare*"),
        ("go2-comparison", "runs/mjlab-smoke/compare/*mjlab-compare*"),
    ]:
        try:
            shots.append((name, compare + [str(latest(pattern))], 5, None))
        except ValueError:
            print("skip", name, "(no run)", flush=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        for port, (name, command, wait, action) in enumerate(shots, start=8101):
            if args.only and name not in args.only:
                continue
            capture(page, name, command, wait, port, action)
        browser.close()


if __name__ == "__main__":
    main()
