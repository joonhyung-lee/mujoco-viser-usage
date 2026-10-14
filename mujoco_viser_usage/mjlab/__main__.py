"""Use mjlab's native training and Viser playback with local task registration.

Examples::

    python -m mujoco_viser_usage.mjlab train Viser-Aloha-Reach-v0 --gpu-ids '[]'
    python -m mujoco_viser_usage.mjlab play Viser-Aloha-Reach-v0 --agent zero
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "play"}:
        raise SystemExit(
            "Usage: python -m mujoco_viser_usage.mjlab {train|play} TASK [mjlab options]"
        )
    command = sys.argv.pop(1)
    from .tasks import register_tasks

    register_tasks()
    if command == "train":
        from mjlab.scripts.train import main as run
    else:
        from mjlab.scripts.play import main as run
    run()


if __name__ == "__main__":
    main()
