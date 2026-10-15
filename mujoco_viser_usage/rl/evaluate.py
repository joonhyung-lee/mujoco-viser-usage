"""Evaluate a saved PPO policy on new seeds and record replayable episodes."""

import argparse
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import TrainConfig
from .storage import append_json, read_json, write_json


def evaluate(checkpoint: Path, run: Path, episodes: int = 3):
    import torch
    from stable_baselines3 import PPO

    from .envs import make_env
    from .storage import EpisodeRecorder

    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    source_config = TrainConfig.load(checkpoint.parent.parent / "config.json")
    config = replace(
        source_config, workers=1, batch_size=source_config.n_steps, seed=1000003
    )
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise ValueError("Evaluation needs a new run directory")
    write_json(run / "config.json", config.to_dict())
    env = None
    steps = 0
    returns = []
    successes = []
    strategies = []
    stop = False
    started = time.time()

    def publish(state):
        write_json(
            run / "status.json",
            {
                "state": state,
                "mode": "evaluation",
                "pid": os.getpid(),
                "worker_pids": [os.getpid()],
                "updated_at": time.time(),
                "started_at": started,
                "steps": steps,
                "run_steps": steps,
                "budget": episodes * config.max_episode_steps,
                "updates": 0,
                "mean_return": float(np.mean(returns)) if returns else None,
                "checkpoint": str(checkpoint),
            },
        )

    try:
        publish("starting")
        env = EpisodeRecorder(make_env(config), run, 0)
        policy = PPO.load(checkpoint, device="cpu")
        held_out = None
        if getattr(config, "configs", 0):
            # Fresh configurations the sampler never saw, balanced over strategies.
            from .configs import sample_configs

            held_out = sample_configs(episodes, config.part, config.seed)
        for episode in range(episodes):
            options = {"config": held_out[episode]} if held_out else None
            obs, _ = env.reset(seed=config.seed + episode, options=options)
            total = 0.0
            while True:
                command = read_json(run / "control.json", {}).get("command", "run")
                while command == "pause":
                    publish("paused")
                    time.sleep(0.1)
                    command = read_json(run / "control.json", {}).get("command", "run")
                if command == "stop":
                    stop = True
                    break
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total += reward
                if steps % 10 == 0:
                    publish("running")
                if terminated or truncated:
                    returns.append(total)
                    successes.append(info["is_success"])
                    strategies.append(info.get("strategy"))
                    append_json(
                        run / "metrics.jsonl",
                        {"steps": steps, "mean_return": float(np.mean(returns))},
                    )
                    break
            if stop:
                break
        report = {
            "episodes": len(returns),
            "returns": returns,
            "successes": successes,
            "strategies": strategies,
            "mean_return": float(np.mean(returns)) if returns else None,
            "success_rate": float(np.mean(successes)) if successes else None,
            "checkpoint": str(checkpoint),
            "seed": config.seed,
        }
        write_json(run / "evaluation.json", report)
        publish("stopped" if stop else "completed")
        return report
    except BaseException as error:
        write_json(
            run / "status.json",
            {
                "state": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": time.time(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(args.checkpoint.resolve(), args.run_dir.resolve(), args.episodes))


if __name__ == "__main__":
    main()
