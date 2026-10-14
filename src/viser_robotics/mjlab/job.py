"""Managed native PPO training process for the Viser dashboard."""

import argparse
import math
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from ..rl.config import MJLAB_STAGES, MjlabTrainConfig, load_config
from ..rl.storage import read_json, write_json
from .runner import RecordedRunner
from .tasks import environment_config, runner_config


class StopRequested(Exception):
    pass


class ManagedEnvironment(RslRlVecEnvWrapper):
    """Control at policy-step boundaries; never rewrite a live physics state."""

    def __init__(self, env, directory, request):
        self.directory = directory
        self.request = request
        self.run_steps = 0
        self.last_status = 0.0
        super().__init__(env)

    def publish(self, state, **extra):
        write_json(
            self.directory / "status.json",
            {
                "backend": "mjlab",
                "state": state,
                "pid": os.getpid(),
                "run_steps": self.run_steps,
                "steps": self.run_steps,
                "total_timesteps": self.request.total_timesteps,
                "updated_at": time.time(),
                **extra,
            },
        )
        self.last_status = time.monotonic()

    def step(self, actions):
        while True:
            command = read_json(self.directory / "control.json", {}).get(
                "command", "run"
            )
            if command == "stop":
                raise StopRequested()
            if command != "pause":
                break
            self.publish("paused")
            time.sleep(0.1)
        result = super().step(actions)
        self.run_steps += self.num_envs
        if time.monotonic() - self.last_status >= 0.5:
            self.publish("running")
        return result


def train(request, directory, resume=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", request.to_dict())
    base = None
    wrapped = None
    runner = None
    try:
        if resume:
            original = load_config(resume.parent / "config.json")
            if (
                not isinstance(original, MjlabTrainConfig)
                or original.task != request.task
            ):
                raise ValueError(
                    "Resume requires a checkpoint from the same mjlab task"
                )
            cfg = pickle.loads((resume.parent / "environment.pkl").read_bytes())
            cfg.scene.num_envs = request.workers
            cfg.seed = request.seed
            agent = read_json(resume.parent / "agent.json")
        else:
            cfg = environment_config(
                MJLAB_STAGES[request.task],
                request.workers,
                request.seed,
                request.terrain,
            )
            agent = asdict(runner_config(MJLAB_STAGES[request.task]))
        agent.update(
            seed=request.seed,
            num_steps_per_env=request.n_steps,
            save_interval=max(
                1, math.ceil(request.save_every / (request.workers * request.n_steps))
            ),
        )
        agent["algorithm"]["learning_rate"] = request.learning_rate
        batch = request.workers * request.n_steps
        agent["algorithm"]["num_mini_batches"] = next(
            n for n in (4, 2, 1) if batch % n == 0
        )
        base = ManagerBasedRlEnv(cfg, device=request.device)
        wrapped = ManagedEnvironment(base, directory, request)
        runner = RecordedRunner(wrapped, agent, str(directory), request.device)
        base.recorder_manager.get_term("rollouts").every = request.record_every
        if resume:
            runner.load(str(resume), map_location=request.device)
        wrapped.publish("running")
        runner.learn(
            math.ceil(request.total_timesteps / batch), init_at_random_ep_len=False
        )
        runner.save(str(directory / "final.pt"))
        wrapped.publish("completed")
    except StopRequested:
        if runner:
            runner.save(str(directory / "stopped.pt"))
        wrapped.publish("stopped")
    except BaseException as error:
        write_json(
            directory / "status.json",
            {
                "backend": "mjlab",
                "state": "failed",
                "error": str(error),
                "run_steps": wrapped.run_steps if wrapped else 0,
                "updated_at": time.time(),
            },
        )
        raise
    finally:
        if base:
            base.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    request = load_config(args.config)
    if not isinstance(request, MjlabTrainConfig):
        parser.error("Expected mjlab request")
    train(request, args.run_dir, args.resume)


if __name__ == "__main__":
    main()
