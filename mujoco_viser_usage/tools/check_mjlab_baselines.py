"""Prove task solvability with physical scripted-controller rollouts."""

import argparse
import json

import torch
from mjlab.envs import ManagerBasedRlEnv

from mujoco_viser_usage.mjlab.baseline import make_controller
from mujoco_viser_usage.mjlab.go2 import base_position
from mujoco_viser_usage.mjlab.mdp import objects
from mujoco_viser_usage.mjlab.tasks import environment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stages", nargs="+", default=["reach", "lift", "place", "bimanual", "go2"]
    )
    args = parser.parse_args()
    for stage in args.stages:
        cfg = environment_config(stage, seed=args.seed)
        cfg.auto_reset = False
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            env.reset(seed=args.seed)
            controller = make_controller(env)
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
                        "objects": objects(env)[0].cpu().tolist()
                        if stage != "go2"
                        else None,
                        "base": base_position(env)[0].cpu().tolist()
                        if stage == "go2"
                        else None,
                    }
                ),
                flush=True,
            )
            if stage == "go2":
                # The open-loop trot is a baseline, not a guaranteed 3 m finisher.
                assert float(base_position(env)[0, 0]) > 0.5, "Go2 trot did not advance"
            else:
                assert success, f"{stage} scripted controller did not complete"
        finally:
            env.close()


if __name__ == "__main__":
    main()
