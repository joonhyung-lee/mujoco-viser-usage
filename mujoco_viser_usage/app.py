"""Command-line entry point. Application behavior lives in ViserParser."""

import argparse

from .scenes import SCENES, resolve_scene
from .viser_parser import ViserParser


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo robotics workspace in Viser")
    parser.add_argument("scene", nargs="?", default="manipulation/ur5e")
    parser.add_argument(
        "--list", action="store_true", help="List scenes by task domain"
    )
    parser.add_argument(
        "--paused", action="store_true", help="Start without advancing physics"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Start the scene sample automatically"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.list:
        for scene in SCENES:
            print(f"{scene.id:26} {scene.description}")
        return
    try:
        scene = resolve_scene(args.scene)
    except ValueError as error:
        parser.error(str(error))
    with ViserParser(
        scene.path,
        host=args.host,
        port=args.port,
        task=scene.task,
        start_running=not args.paused,
        demo=args.demo,
        description=f"{scene.id} · {scene.description}",
    ) as robot:
        robot.run()


if __name__ == "__main__":
    main()
