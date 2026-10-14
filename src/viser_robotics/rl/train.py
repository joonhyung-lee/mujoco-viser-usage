"""PPO training with spawned MuJoCo workers, checkpoints and cooperative control."""

import argparse
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .config import TrainConfig
from .storage import append_json, new_run, read_json, write_json


def make_worker(config, run, rank):
    # Imports inside the child keep the UI/supervisor independent of torch.
    from stable_baselines3.common.monitor import Monitor

    from .envs import RobotEnv
    from .storage import EpisodeRecorder

    return Monitor(EpisodeRecorder(RobotEnv(config), Path(run), rank))


def train(config: TrainConfig, run: Path, resume: Path | None = None):
    from functools import partial

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(1)
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise ValueError(
            "Run directory already contains an experiment; use a new run for resume"
        )
    write_json(run / "config.json", config.to_dict())
    if not (run / "control.json").exists():
        write_json(run / "control.json", {"command": "run"})
    write_json(
        run / "status.json",
        {
            "state": "starting",
            "pid": os.getpid(),
            "updated_at": time.time(),
            "steps": 0,
        },
    )
    vector = None
    started = time.time()

    class Observer(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_publish = 0.0
            self.last_save = 0
            self.stopped = False
            self.initial_steps = 0
            self.pids = []
            self.updates = 0

        def _on_training_start(self):
            self.initial_steps = self.model.num_timesteps
            self.pids = self.training_env.get_attr("worker_pid")
            self.publish("running")

        def publish(self, state):
            from math import isfinite

            self.updates = int(self.model._n_updates)
            metrics = {
                key: float(value)
                for key, value in self.model.logger.name_to_value.items()
                if key.startswith("train/")
                and isinstance(value, (float, int))
                and isfinite(value)
            }
            episodes = list(self.model.ep_info_buffer)
            elapsed = time.time() - started
            write_json(
                run / "status.json",
                {
                    "state": state,
                    "pid": os.getpid(),
                    "worker_pids": self.pids,
                    "updated_at": time.time(),
                    "started_at": started,
                    "steps": self.model.num_timesteps,
                    "run_steps": self.model.num_timesteps - self.initial_steps,
                    "budget": config.total_timesteps,
                    "updates": self.updates,
                    "elapsed": elapsed,
                    "fps": (self.model.num_timesteps - self.initial_steps)
                    / max(elapsed, 0.001),
                    "mean_return": sum(x["r"] for x in episodes) / len(episodes)
                    if episodes
                    else None,
                    "metrics": metrics,
                    "resume": str(resume) if resume else None,
                },
            )
            self.last_publish = time.monotonic()

        def save(self, name):
            checkpoint = run / "checkpoints" / name
            checkpoint.parent.mkdir(exist_ok=True)
            temporary = checkpoint.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                self.model.save(stream)
            os.replace(temporary, checkpoint.with_suffix(".zip"))
            append_json(
                run / "events.jsonl",
                {
                    "event": "checkpoint",
                    "path": str(checkpoint.with_suffix(".zip").relative_to(run)),
                    "steps": self.model.num_timesteps,
                    "time": time.time(),
                },
            )
            self.last_save = self.model.num_timesteps

        def _on_step(self):
            command = read_json(run / "control.json", {}).get("command", "run")
            while command == "pause":
                if time.monotonic() - self.last_publish > 0.5:
                    self.publish("paused")
                time.sleep(0.05)
                command = read_json(run / "control.json", {}).get("command", "run")
            if command == "stop":
                self.stopped = True
                self.publish("stopping")
                return False
            if time.monotonic() - self.last_publish > 0.25:
                self.publish("running")
            if self.model.num_timesteps - self.last_save >= config.save_every:
                self.save(f"step-{self.model.num_timesteps:09d}")
            return True

        def _on_rollout_end(self):
            # PPO updates after this callback; published train metrics are the prior completed update.
            self.updates = int(self.model._n_updates)
            self.publish("running")
            append_json(run / "metrics.jsonl", read_json(run / "status.json"))

    observer = Observer()
    try:
        if resume:
            old = TrainConfig.load(resume.parent.parent / "config.json")
            if old.task != config.task:
                raise ValueError("Resume checkpoint task must match the new experiment")
        vector = SubprocVecEnv(
            [partial(make_worker, config, str(run), i) for i in range(config.workers)],
            start_method="spawn",
        )
        if resume:
            model = PPO.load(
                resume,
                env=vector,
                device="cpu",
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
            )
        else:
            model = PPO(
                "MlpPolicy",
                vector,
                seed=config.seed,
                device="cpu",
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=5,
                policy_kwargs={"net_arch": [64, 64]},
                verbose=0,
            )
        observer.init_callback(model)
        observer.save("initial")
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=observer,
            reset_num_timesteps=not bool(resume),
        )
        observer.updates = int(model._n_updates)
        observer.save("final")
        observer.publish("stopped" if observer.stopped else "completed")
        append_json(run / "metrics.jsonl", read_json(run / "status.json"))
    except BaseException as error:
        write_json(
            run / "status.json",
            {
                "state": "failed",
                "pid": os.getpid(),
                "updated_at": time.time(),
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if vector is not None:
            vector.close()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--task", choices=["go2_velocity", "aloha_reach", "aloha_pick_place"]
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--runs", type=Path, default=Path("rl/runs"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    updates = {
        k: v
        for k, v in {
            "task": args.task,
            "workers": args.workers,
            "total_timesteps": args.steps,
        }.items()
        if v is not None
    }
    config = replace(config, **updates)
    run = args.run_dir or new_run(args.runs, config.task)
    print(f"RUN_DIR={run.resolve()}", flush=True)
    train(config, run, args.resume)


if __name__ == "__main__":
    main()
