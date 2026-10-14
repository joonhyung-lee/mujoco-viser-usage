"""Validated, serializable experiment definitions."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..scenes import TERRAINS, resolve_scene

TASK_SCENES = {
    "aloha_reach": "manipulation/aloha",
    "aloha_pick_place": "manipulation/aloha",
    "go2_velocity": "locomotion/go2",
}


@dataclass(frozen=True)
class TrainConfig:
    task: str = "go2_velocity"
    terrain: str = "flat"
    seed: int = 7
    workers: int = 4
    total_timesteps: int = 16384
    n_steps: int = 256
    batch_size: int = 256
    learning_rate: float = 0.0003
    max_episode_steps: int = 400
    control_dt: float = 0.02
    save_every: int = 4096
    record_every: int = 5

    def __post_init__(self):
        for name in (
            "seed",
            "workers",
            "total_timesteps",
            "n_steps",
            "batch_size",
            "max_episode_steps",
            "save_every",
            "record_every",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.terrain not in TERRAINS:
            raise ValueError(f"Unknown terrain: {self.terrain}")
        if self.task != "go2_velocity" and self.terrain != "flat":
            raise ValueError("Terrain variants apply only to Go2 locomotion")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.task not in TASK_SCENES:
            raise ValueError(f"Unknown RL task: {self.task}")
        if not 1 <= self.workers <= 16:
            raise ValueError("workers must be between 1 and 16")
        if self.n_steps < 8 or self.total_timesteps < 1 or self.max_episode_steps < 2:
            raise ValueError("Invalid rollout or episode length")
        if not 2 <= self.batch_size <= self.workers * self.n_steps:
            raise ValueError("batch_size must fit the combined worker rollout")
        if (self.workers * self.n_steps) % self.batch_size:
            raise ValueError("batch_size must divide workers × n_steps")
        if not 0.002 <= self.control_dt <= 0.1 or not 0 < self.learning_rate <= 0.01:
            raise ValueError("Invalid control_dt or learning_rate")
        if self.save_every < 1 or self.record_every < 1:
            raise ValueError("Recording/checkpoint intervals must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))


def scene_path(config: TrainConfig):
    scene_id = TASK_SCENES[config.task]
    if config.terrain != "flat":
        scene_id += "_" + config.terrain
    return resolve_scene(scene_id).path


MJLAB_STAGES = {
    "aloha_reach": "reach",
    "aloha_pick_place": "bimanual",
    "go2_velocity": "go2",
}


@dataclass(frozen=True)
class MjlabTrainConfig:
    """Dashboard request; imported without loading Torch, Warp, or mjlab."""

    backend: str = "mjlab"
    task: str = "aloha_reach"
    terrain: str = "flat"
    seed: int = 7
    workers: int = 4
    total_timesteps: int = 16384
    n_steps: int = 24
    learning_rate: float = 0.0003
    save_every: int = 4096
    record_every: int = 5
    device: str = "cpu"

    def __post_init__(self):
        if self.backend != "mjlab" or self.task not in MJLAB_STAGES:
            raise ValueError(
                "mjlab backend supports ALOHA reach, pick/place, and Go2 velocity"
            )
        if self.terrain not in TERRAINS:
            raise ValueError(f"Unknown terrain: {self.terrain}")
        if self.task != "go2_velocity" and self.terrain != "flat":
            raise ValueError("ALOHA uses its tabletop workspace")
        for name in (
            "seed",
            "workers",
            "total_timesteps",
            "n_steps",
            "save_every",
            "record_every",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid {name}")
        if self.workers > 8192 or self.n_steps < 8:
            raise ValueError("Use 1–8192 environments and at least 8 rollout steps")
        if not 0 < self.learning_rate <= 0.01:
            raise ValueError("Invalid learning rate")
        if not isinstance(self.device, str) or (
            self.device != "cpu"
            and not (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("Device must be cpu or cuda:N")

    def to_dict(self):
        return asdict(self)


def parse_config(payload):
    backend = payload.get("backend", "sb3")
    if backend == "mjlab":
        return MjlabTrainConfig(**payload)
    if backend != "sb3":
        raise ValueError(f"Unknown training backend: {backend}")
    return TrainConfig(
        **{key: value for key, value in payload.items() if key != "backend"}
    )


def load_config(path):
    return parse_config(json.loads(Path(path).read_text()))
