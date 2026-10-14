"""Bounded rollout capture through mjlab's terminal-state lifecycle hooks."""

import os
import time
from pathlib import Path

import numpy as np
from mjlab.managers.recorder_manager import RecorderTerm

from ..rl.storage import append_json, read_json, write_json, write_npz


def stage_of(env):
    return env.cfg.events["reset"].params["stage"]


def task_fields(env, world):
    """Task-specific trace fields (targets, contacts, tracked positions)."""
    if stage_of(env) == "go2":
        from .go2 import snapshot
    else:
        from .mdp import snapshot
    return snapshot(env, world)


class RolloutRecorder(RecorderTerm):
    """Record a few worlds, including terminal states before automatic reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.root = Path(cfg.params["directory"])
        self.count = min(int(cfg.params.get("worlds", 4)), env.num_envs)
        self.every = int(cfg.params.get("every", 5))
        if self.count < 1 or self.every < 1:
            raise ValueError("Recording worlds and interval must be positive")
        self.frames = [[] for _ in range(self.count)]
        self.episodes = [-1] * self.count
        self.returns = [0.0] * self.count
        self.steps = [0] * self.count
        self.started = [0.0] * self.count

    def _directory(self, world):
        return self.root / "workers" / str(world)

    def _snapshot(self, world, reward=0.0):
        env = self._env
        data = env.sim.data
        names = ["qpos", "qvel", "ctrl", "qfrc_applied"]
        if env.sim.mj_model.nmocap:
            # Warp cannot expose empty mocap arrays to Torch.
            names += ["mocap_pos", "mocap_quat"]
        frame = {
            name: getattr(data, name)[world].detach().cpu().numpy().copy()
            for name in names
        }
        frame.update(
            time=np.array(self.steps[world] * env.step_dt),
            action=env.action_manager.action[world].cpu().numpy().copy(),
            reward=np.array(reward),
            **task_fields(env, world),
        )
        frame["reward_terms"] = np.asarray(
            [
                float(value[0]) if self.steps[world] else 0.0
                for _, value in env.reward_manager.get_active_iterable_terms(world)
            ]
        )
        self.frames[world].append(frame)
        write_npz(self._directory(world) / "latest.npz", **frame)
        write_json(
            self._directory(world) / "state.json",
            {
                "worker": world,
                "pid": os.getpid(),
                "episode": self.episodes[world],
                "steps": self.steps[world],
                "return": self.returns[world],
                "state": "running",
                "sim_time": float(frame["time"]),
                "updated_at": time.time(),
            },
        )

    def record_post_reset(self, env_ids):
        for world in env_ids.cpu().tolist():
            if world >= self.count:
                continue
            if self.frames[world] and self.steps[world]:
                self._finish(world, complete=False)
            self.episodes[world] += 1
            self.frames[world] = []
            self.returns[world] = 0.0
            self.steps[world] = 0
            self.started[world] = time.time()
            self._snapshot(world)

    def _transition(self, world, terminal):
        reward = float(self._env.reward_buf[world])
        self.steps[world] += 1
        self.returns[world] += reward
        if terminal or self.steps[world] % self.every == 0:
            self._snapshot(world, reward)
        if terminal:
            self._finish(world, complete=True)

    def record_pre_reset(self, env_ids):
        for world in env_ids.cpu().tolist():
            if world < self.count and self.frames[world]:
                self._transition(world, terminal=True)

    def record_post_step(self):
        for world in range(self.count):
            if not self.frames[world]:
                continue
            if self._env.cfg.auto_reset and bool(self._env.reset_buf[world]):
                continue
            self._transition(world, terminal=bool(self._env.reset_buf[world]))

    def _finish(self, world, complete):
        frames = self.frames[world]
        if not frames:
            return
        env = self._env
        trace = f"episode-{self.episodes[world]:06d}.npz"
        write_npz(
            self._directory(world) / trace,
            **{key: np.stack([frame[key] for frame in frames]) for key in frames[0]},
        )
        success = (
            bool(env.termination_manager.get_term("success")[world])
            if complete
            else False
        )
        append_json(
            self._directory(world) / "episodes.jsonl",
            {
                "worker": world,
                "episode": self.episodes[world],
                "trace": trace,
                "seed": env.cfg.seed,
                "return": self.returns[world],
                "length": self.steps[world],
                "is_success": success,
                "state": "complete" if complete else "interrupted",
                "terminated": bool(env.reset_terminated[world]) if complete else False,
                "truncated": bool(env.reset_time_outs[world]) if complete else False,
                "started_at": self.started[world],
                "finished_at": time.time(),
            },
        )
        self.frames[world] = []

    def close(self):
        for world in range(self.count):
            if self.frames[world] and self.steps[world]:
                # Preserve the exact final state even between recording intervals.
                if self.steps[world] % self.every:
                    self._snapshot(world, float(self._env.reward_buf[world]))
                self._finish(world, complete=False)
            current = read_json(self._directory(world) / "state.json", {})
            if current:
                current.update(state="closed", updated_at=time.time())
                write_json(self._directory(world) / "state.json", current)
