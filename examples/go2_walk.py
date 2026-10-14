"""Run: uv run python examples/go2_walk.py"""

from viser_robotics import ViserParser

with ViserParser.from_scene("locomotion/go2", demo=True) as robot:
    robot.run()
