"""Run from the repository root: uv run python examples/manual_loop.py."""

import time

from viser_robotics import ViserParser

with ViserParser(port=8081) as robot:
    robot.set_control("shoulder_pan", -1.3)
    try:
        while True:
            robot.step(nstep=10)
            robot.render()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
