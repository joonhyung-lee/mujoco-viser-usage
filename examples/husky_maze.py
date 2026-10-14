"""Run: uv run python examples/husky_maze.py"""

from viser_robotics import ViserParser

with ViserParser.from_scene("navigation/husky", demo=True) as robot:
    robot.run()
