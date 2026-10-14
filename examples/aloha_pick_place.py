"""Run: uv run python examples/aloha_pick_place.py"""

from viser_robotics import ViserParser

with ViserParser.from_scene("manipulation/aloha", demo=True) as robot:
    robot.run()
