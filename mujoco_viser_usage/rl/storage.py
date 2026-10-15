"""Atomic snapshots and append-only episode records, one writer per file."""

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import gymnasium as gym
import numpy as np


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def append_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A reader may see the final line before its writer finishes.
            continue
    return records


def write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def new_run(root: Path, task: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{task}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True)
    return path


class EpisodeRecorder(gym.Wrapper):
    def __init__(self, env, run: Path, worker: int):
        super().__init__(env)
        self.run = Path(run)
        self.worker = worker
        self.worker_pid = os.getpid()
        self.directory = self.run / "workers" / str(worker)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.episode_index = -1
        self.recording = False
        self.frames = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        if self.recording and self.steps:
            self._finish(False, True, {"is_success": False}, "interrupted")
        self.episode_index += 1
        self.seed_value = int(
            seed
            if seed is not None
            else self.unwrapped.config.seed + 10000 * self.worker + self.episode_index
        )
        obs, info = self.env.reset(seed=self.seed_value, options=options)
        self.frames = []
        self.return_value = 0.0
        self.steps = 0
        self.recording = True
        self.started_at = time.time()
        self._snapshot("running")
        return obs, info

    def _snapshot(self, state):
        env = self.unwrapped
        frame = {
            "qpos": env.data.qpos.copy(),
            "qvel": env.data.qvel.copy(),
            "ctrl": env.data.ctrl.copy(),
            "qfrc_applied": env.data.qfrc_applied.copy(),
            "time": np.array(env.data.time),
            "targets": env.targets.copy(),
            "command": np.array(env.command),
        }
        self.frames.append(frame)
        write_npz(self.directory / "latest.npz", **frame)
        write_json(
            self.directory / "state.json",
            {
                "worker": self.worker,
                "pid": self.worker_pid,
                "episode": self.episode_index,
                "seed": self.seed_value,
                "steps": self.steps,
                "return": self.return_value,
                "state": state,
                "updated_at": time.time(),
                "sim_time": float(env.data.time),
            },
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.return_value += reward
        if (
            self.steps % self.unwrapped.config.record_every == 0
            or terminated
            or truncated
        ):
            self._snapshot("episode_end" if terminated or truncated else "running")
        if terminated or truncated:
            self._finish(terminated, truncated, info, "complete")
        return obs, reward, terminated, truncated, info

    def _finish(self, terminated, truncated, info, state):
        if self.frames and float(self.frames[-1]["time"]) != float(
            self.unwrapped.data.time
        ):
            self._snapshot(state)
        trace_name = f"episode-{self.episode_index:06d}.npz"
        if self.frames:
            write_npz(
                self.directory / trace_name,
                **{
                    key: np.stack([frame[key] for frame in self.frames])
                    for key in self.frames[0]
                },
            )
        append_json(
            self.directory / "episodes.jsonl",
            {
                "worker": self.worker,
                "episode": self.episode_index,
                "seed": self.seed_value,
                "return": self.return_value,
                "length": self.steps,
                "is_success": bool(info.get("is_success", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "state": state,
                "started_at": self.started_at,
                "finished_at": time.time(),
                "distance": float(info.get("distance", 0.0)),
                "config": int(info.get("config", -1)),
                "strategy": info.get("strategy"),
                "trace": trace_name,
            },
        )
        self.recording = False

    def close(self):
        if self.recording and self.steps:
            self._finish(False, True, {"is_success": False}, "interrupted")
        current = read_json(self.directory / "state.json", {})
        current.update(state="closed", updated_at=time.time())
        write_json(self.directory / "state.json", current)
        self.env.close()
