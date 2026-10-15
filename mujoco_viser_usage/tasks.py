"""Task lifecycle shared by scripted examples and the interactive controls."""

from typing import Protocol

from .simulation import Simulation


class Task(Protocol):
    label: str
    active: bool
    status: str

    def start(self) -> None: ...
    def reset(self) -> None: ...
    def update(self) -> None: ...


def create_task(name: str | None, simulation: Simulation) -> Task | None:
    if name is None:
        return None
    if name == "pick_place":
        from .pick_place import PickPlaceTask

        return PickPlaceTask(simulation)
    if name == "navigation":
        from .scenes.navigation import MazeNavigationTask

        return MazeNavigationTask(simulation)
    if name == "aloha":
        from .scenes.aloha_task import AlohaTask

        return AlohaTask(simulation)
    if name in ("franka_plug", "franka_cable", "franka_nist"):
        from .scenes.franka_task import FrankaTask

        return FrankaTask(simulation, name.removeprefix("franka_"))
    if name == "go2_walk":
        from .scenes.go2_task import Go2WalkTask

        return Go2WalkTask(simulation)
    raise ValueError(f"Unknown task: {name}")
