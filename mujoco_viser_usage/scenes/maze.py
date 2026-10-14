"""Shared occupancy map: one definition for collision walls and route planning.

Following OGBench's robot/environment separation, map coordinates are independent
of the robot. This layout is original; it is not an OGBench benchmark task.
"""

from collections import deque

import numpy as np

LAYOUT = (
    "#######",
    "#S....#",
    "###.#.#",
    "#...#.#",
    "#.###.#",
    "#....G#",
    "#######",
)
CELL_SIZE = 2.0


def world(cell: tuple[int, int]) -> np.ndarray:
    row, col = cell
    return np.array([(col - 1) * CELL_SIZE, (row - 1) * CELL_SIZE])


def route() -> np.ndarray:
    """Shortest four-connected path, including start and goal cell centers."""
    start = next(
        (r, c) for r, line in enumerate(LAYOUT) for c, v in enumerate(line) if v == "S"
    )
    goal = next(
        (r, c) for r, line in enumerate(LAYOUT) for c, v in enumerate(line) if v == "G"
    )
    frontier = deque([start])
    previous = {start: None}
    while frontier:
        cell = frontier.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(world(cell))
                cell = previous[cell]
            return np.asarray(path[::-1])
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            neighbor = cell[0] + dr, cell[1] + dc
            r, c = neighbor
            if (
                0 <= r < len(LAYOUT)
                and 0 <= c < len(LAYOUT[r])
                and LAYOUT[r][c] != "#"
                and neighbor not in previous
            ):
                previous[neighbor] = cell
                frontier.append(neighbor)
    raise ValueError("Maze goal is unreachable")
