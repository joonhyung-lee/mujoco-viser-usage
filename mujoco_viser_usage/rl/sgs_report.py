"""Summarize insertion runs: held-out success per start strategy, with Wilson 95% CIs.

Usage: uv run --extra rl python -m mujoco_viser_usage.rl.sgs_report runs/sgs/*-eval
"""

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, center - half, center + half


def summarize(run: Path):
    report = json.loads((run / "evaluation.json").read_text())
    config = json.loads((run / "config.json").read_text())
    per = defaultdict(lambda: [0, 0])
    for strategy, success in zip(report["strategies"], report["successes"]):
        per[strategy][0] += success
        per[strategy][1] += 1
    return config, report, per


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print("| run | part | sampler | all | near goal | hovering | reaching |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for run in args.runs:
        config, report, per = summarize(run)
        k = sum(report["successes"])
        n = len(report["successes"])
        _, lo, hi = wilson(k, n)
        cells = [f"{k}/{n} ({lo:.2f}–{hi:.2f})"]
        for strategy in ("near_goal", "stable", "reaching"):
            s, m = per.get(strategy, (0, 0))
            cells.append(f"{s}/{m}" if m else "—")
        print(
            f"| {run.name} | {config['part']} | {config.get('sampler', '—')} | "
            + " | ".join(cells)
            + " |"
        )


if __name__ == "__main__":
    main()
