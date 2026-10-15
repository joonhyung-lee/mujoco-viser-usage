"""Fixed task-configuration buffers for insertion, after Success-Guided Sampling (SGS).

A configuration is one reset: which part, which reset strategy, and the hidden
offsets. Buffers are generated from a seed, so every worker and the sampler in
the training process rebuild the same list from the TrainConfig.
"""

from dataclasses import astuple, dataclass

import numpy as np

STRATEGIES = ("near_goal", "stable", "reaching")
# Reaching places the part anywhere on this table patch, left of the connector row.
WORKSPACE_X = (0.28, 0.72)
WORKSPACE_Y = (-0.34, 0.26)


@dataclass(frozen=True)
class InsertConfig:
    strategy: str
    hole_x: float  # hidden hole offset, m
    hole_y: float
    in_hand_x: float  # part offset along the pad axis normal, m
    yaw: float  # gripper yaw, rad
    twist: float  # part yaw relative to the gripper (square pegs), rad
    fraction: float  # near_goal: 0 seated .. 1.1 above the stable height
    place_x: float  # reaching: part position on the table
    place_y: float
    aperture: float  # reaching: finger opening 0 (closed on the part) .. 1 (fully open)

    def as_array(self):
        return np.array(
            [STRATEGIES.index(self.strategy), *astuple(self)[1:]], dtype=np.float32
        )


def sample_config(rng, part, strategy):
    square = part.startswith("peg")
    return InsertConfig(
        strategy,
        *rng.uniform(-0.005, 0.005, 2),
        float(rng.uniform(-0.003, 0.003)),
        float(rng.uniform(-0.17, 0.17)) if square else 0.0,
        float(rng.uniform(-0.09, 0.09)) if square else 0.0,
        float(rng.uniform(0.0, 1.1)),
        float(rng.uniform(*WORKSPACE_X)),
        float(rng.uniform(*WORKSPACE_Y)),
        float(rng.uniform(0.0, 1.0)),
    )


def sample_configs(count, part, seed, strategies=STRATEGIES):
    """`count` configurations balanced over the strategies, deterministic in seed."""
    rng = np.random.default_rng(seed)
    return [
        sample_config(rng, part, strategies[i % len(strategies)]) for i in range(count)
    ]
