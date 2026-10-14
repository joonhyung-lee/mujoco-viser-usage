"""Read-only, simulation-time playback. Never writes into a training worker."""

from pathlib import Path

import mujoco
import numpy as np

from .storage import read_jsonl


def sample_trace(model, trace, position):
    """Interpolate within one episode, including free/ball joint quaternion slerp."""
    times = trace["time"]
    position = float(np.clip(position, times[0], times[-1]))
    right = min(int(np.searchsorted(times, position, side="right")), len(times) - 1)
    left = max(0, right - 1)
    duration = float(times[right] - times[left])
    alpha = (position - times[left]) / duration if duration > 0 else 0.0
    frame = {
        key: (1 - alpha) * value[left] + alpha * value[right]
        for key, value in trace.items()
    }
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(
        model, velocity, 1.0, trace["qpos"][left], trace["qpos"][right]
    )
    frame["qpos"] = np.array(trace["qpos"][left], dtype=np.float64, copy=True)
    mujoco.mj_integratePos(model, frame["qpos"], velocity, alpha)
    frame["time"] = np.array(position)
    return frame


class RolloutPlayer:
    """Play newest completed trace at 1x; switch episodes only at boundaries.

    Training can run faster than rendering. Drop superseded episodes, never
    skip states within the episode being shown or interpolate across resets.
    """

    def __init__(self, model):
        self.model = model
        self.path = None
        self.trace = None
        self.started = 0.0
        self.checked = -float("inf")
        self.finished = False

    def sample(self, directory: Path, now: float, *, loop=False):
        if (self.trace is None or self.finished) and now - self.checked >= 0.5:
            self.checked = now
            records = read_jsonl(directory / "episodes.jsonl")
            complete = [
                record for record in records if record.get("state") == "complete"
            ]
            paths = (
                [directory / record["trace"] for record in complete]
                if complete
                else sorted(directory.glob("episode-*.npz"))
            )
            if paths and paths[-1] != self.path:
                with np.load(paths[-1], allow_pickle=False) as archive:
                    self.trace = dict(archive)
                self.path = paths[-1]
                self.started = now
                self.finished = False
        if self.trace is None:
            return None
        duration = float(self.trace["time"][-1] - self.trace["time"][0])
        if self.finished and loop and now - self.started > duration + 0.6:
            self.started = now
            self.finished = False
        position = self.trace["time"][0] + now - self.started
        self.finished = position >= self.trace["time"][-1]
        return sample_trace(self.model, self.trace, position)


class PlaybackClock:
    """A pausable display clock; does not pause PPO workers."""

    def __init__(self):
        self.value = 0.0
        self.previous = None
        self.paused = False

    def advance(self, now, speed=1.0):
        if self.previous is not None and not self.paused:
            self.value += max(0.0, now - self.previous) * speed
        self.previous = now
        return self.value
