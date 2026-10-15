"""Bundled scenes grouped by task domain; no training datasets are downloaded."""

from dataclasses import dataclass
from pathlib import Path

from ..simulation import PICK_PLACE_SCENE

ROOT = Path(__file__).resolve().parents[2] / "assets"


@dataclass(frozen=True)
class Scene:
    id: str
    path: Path
    description: str
    task: str | None = None


SCENES = (
    Scene(
        "manipulation/ur5e",
        PICK_PLACE_SCENE,
        "UR5e + RG2 · contact-based pick & place",
        "pick_place",
    ),
    Scene(
        "manipulation/franka",
        ROOT / "manipulation/franka/workspace.xml",
        "Franka Panda · tabletop objects · manual joints",
    ),
    Scene(
        "manipulation/franka_plug",
        ROOT / "manipulation/franka/plug.xml",
        "Franka Panda · plug insertion · 2.5 mm clearance",
        "franka_plug",
    ),
    Scene(
        "manipulation/franka_cable",
        ROOT / "manipulation/franka/cable.xml",
        "Franka Panda · cable-end routing into a clip",
        "franka_cable",
    ),
    Scene(
        "manipulation/nist_board",
        ROOT / "manipulation/franka/nist_board.xml",
        "Franka Panda · NIST Assembly Task Board #1 · pegs, gears, nuts, connectors",
        "franka_nist",
    ),
    Scene(
        "manipulation/aloha",
        ROOT / "manipulation/aloha/workspace.xml",
        "ALOHA · two arms · contact-based pick & place",
        "aloha",
    ),
    Scene(
        "locomotion/go1",
        ROOT / "locomotion/go1/go1_maze.xml",
        "Unitree Go1 · maze · standing position servos",
    ),
    Scene(
        "locomotion/go2",
        ROOT / "locomotion/go2/go2_maze.xml",
        "Unitree Go2 · maze · scripted trot sample",
        "go2_walk",
    ),
    Scene(
        "locomotion/spot",
        ROOT / "locomotion/spot/spot_maze.xml",
        "Spot · maze · standing position servos",
    ),
    Scene(
        "locomotion/spot_arm",
        ROOT / "locomotion/spot/spot_arm_maze.xml",
        "Spot + arm · maze · standing position servos",
    ),
    Scene(
        "navigation/husky",
        ROOT / "navigation/husky/maze.xml",
        "Husky MJCF + CAD meshes · four-wheel skid steering",
        "navigation",
    ),
    Scene(
        "navigation/turtlebot3",
        ROOT / "navigation/turtlebot3/maze.xml",
        "TurtleBot3 approximation · differential drive",
        "navigation",
    ),
)


TERRAINS = ("flat", "ramps", "steps", "rough", "mixed")
SCENES += tuple(
    Scene(
        f"{domain}/{robot}_{terrain}",
        ROOT / domain / robot / f"{terrain}.xml",
        f"{robot.title()} · {terrain} contact terrain",
        task,
    )
    for domain, robot, task in (
        ("navigation", "husky", "navigation"),
        ("locomotion", "go2", "go2_walk"),
    )
    for terrain in TERRAINS[1:]
)


def resolve_scene(name: str) -> Scene:
    for scene in SCENES:
        if scene.id == name:
            return scene
    path = Path(name).expanduser()
    if path.is_file():
        # Preserve behavior for callers passing the original UR5e XML path.
        for scene in SCENES:
            if path.resolve() == scene.path.resolve():
                return scene
        return Scene(path.stem, path, "Custom MuJoCo XML")
    raise ValueError(f"Unknown scene: {name}. Use --list to see available scenes.")
