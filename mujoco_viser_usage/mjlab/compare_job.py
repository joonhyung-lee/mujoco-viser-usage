"""Dashboard-owned same-seed evaluation process."""

import argparse
import time
from pathlib import Path

from ..rl.storage import read_json, write_json
from .evaluate import evaluate
from .job import StopRequested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir

    def control():
        while True:
            command = read_json(output / "control.json", {}).get("command", "run")
            if command == "stop":
                raise StopRequested()
            if command != "pause":
                return
            time.sleep(0.1)

    try:
        write_json(
            output / "status.json",
            {"state": "running", "backend": "mjlab", "updated_at": time.time()},
        )
        evaluate(
            args.checkpoint.parent,
            args.checkpoint,
            output,
            [1000003, 1000004, 1000005],
            control=control,
        )
        write_json(
            output / "status.json",
            {"state": "completed", "backend": "mjlab", "updated_at": time.time()},
        )
    except StopRequested:
        write_json(
            output / "status.json", {"state": "stopped", "updated_at": time.time()}
        )
    except BaseException as error:
        write_json(
            output / "status.json",
            {"state": "failed", "error": str(error), "updated_at": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
