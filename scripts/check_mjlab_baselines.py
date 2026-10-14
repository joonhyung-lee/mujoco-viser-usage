"""Prove ALOHA task solvability with physical scripted-controller rollouts."""

import argparse
import json

import torch
from mjlab.envs import ManagerBasedRlEnv

from viser_robotics.mjlab.baseline import ScriptedController
from viser_robotics.mjlab.mdp import objects
from viser_robotics.mjlab.tasks import environment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for stage in ("reach", "lift", "place", "bimanual"):
        cfg = environment_config(stage, seed=args.seed)
        cfg.auto_reset = False
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            env.reset(seed=args.seed)
            controller = ScriptedController(env)
            with torch.inference_mode():
                for step in range(env.max_episode_length):
                    observation, _, terminated, timeout, _ = env.step(controller())
                    assert torch.isfinite(observation["actor"]).all(), stage
                    if bool(terminated[0] | timeout[0]):
                        break
            success = bool(env.termination_manager.get_term("success")[0])
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "seed": args.seed,
                        "success": success,
                        "steps": step + 1,
                        "objects": objects(env)[0].cpu().tolist(),
                    }
                ),
                flush=True,
            )
            assert success, f"{stage} scripted controller did not complete"
        finally:
            env.close()


if __name__ == "__main__":
    main()
