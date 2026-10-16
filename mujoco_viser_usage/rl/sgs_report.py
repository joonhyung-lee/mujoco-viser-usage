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


def grouped(runs):
    """One row per sampler: per-seed held-out successes and the pooled interval."""
    pooled = defaultdict(lambda: {"seeds": [], "per": defaultdict(lambda: [0, 0])})
    for run in runs:
        config, report, per = summarize(run)
        entry = pooled[config.get("sampler", "—")]
        entry["seeds"].append(sum(report["successes"]))
        for strategy, (k, n) in per.items():
            entry["per"][strategy][0] += k
            entry["per"][strategy][1] += n
    print(
        "| sampler | seeds | per seed /60 | pooled | near goal | hovering | reaching |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for sampler, entry in pooled.items():
        k = sum(entry["seeds"])
        n = 60 * len(entry["seeds"])
        p, lo, hi = wilson(k, n)
        cells = [f"{p:.2f} ({lo:.2f}–{hi:.2f})"]
        for strategy in ("near_goal", "stable", "reaching"):
            s, m = entry["per"].get(strategy, (0, 0))
            cells.append(f"{s / m:.2f}" if m else "—")
        print(
            f"| {sampler} | {len(entry['seeds'])} | {', '.join(map(str, entry['seeds']))} | "
            + " | ".join(cells)
            + " |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--group", action="store_true", help="pool seeds per sampler")
    args = parser.parse_args()
    if args.group:
        return grouped(args.runs)
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
