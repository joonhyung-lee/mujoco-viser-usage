"""Same-seed scripted/policy evaluation with terminal traces and diagnostics."""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderManager, RecorderTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from ..rl.storage import new_run, read_json, read_jsonl, write_json
from .assets import export_scene
from .baseline import make_controller
from .recording import RolloutRecorder, task_fields


def diagnostics(trace, stage):
    if stage == "go2":
        from .go2 import diagnostics as go2_diagnostics

        return go2_diagnostics(trace)
    count = 2 if stage in {"reach", "bimanual"} else 1
    obj = trace["object_pos"][:, :count]
    tools = trace["eef_pos"][:, :count]
    goals = trace["targets"][:, :count]
    distance = np.linalg.norm(
        (tools if stage == "reach" else obj) - goals, axis=-1
    ).mean(-1)
    height = obj[:, :, 2].max(-1)
    force = np.linalg.norm(trace["contact_force"], axis=-1).max(axis=(1, 2))
    penetration = np.maximum(0, -trace["contact_dist"].min(axis=(1, 2)))
    lifted = height > 0.10
    slipped = np.maximum.accumulate(lifted) & (height < 0.06)
    events = []
    for kind, mask in [
        ("lift", lifted),
        ("lowered_after_lift", slipped),
        ("penetration", penetration > 0.003),
    ]:
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        events.extend({"kind": kind, "time": float(trace["time"][i])} for i in starts)
    # "Lowered after lift" is an observation; intentional placement is not labeled a slip.
    return {
        "final_goal_error_m": float(distance[-1]),
        "max_object_height_m": float(height.max()),
        "max_recorded_contact_force_n": float(force.max()),
        "max_recorded_penetration_m": float(penetration.max()),
        "events": sorted(events, key=lambda event: event["time"]),
    }


def evaluate(
    run: Path, checkpoint: Path, output: Path, seeds, device="cpu", control=None
):
    manifest = read_json(run / "run.json")
    if not manifest or manifest.get("backend") != "mjlab":
        raise ValueError("Expected a recorded mjlab run")
    reports = []
    for seed in seeds:
        initial_states = []
        for controller in ("scripted", "policy"):
            # These are locally generated experiment artifacts, like the checkpoint.
            cfg = pickle.loads((run / "environment.pkl").read_bytes())
            cfg.scene.num_envs = 1
            cfg.auto_reset = False
            cfg.seed = seed
            cfg.recorders = {}
            env = ManagerBasedRlEnv(cfg, device=device)
            directory = output / f"seed-{seed}" / controller
            try:
                wrapped = RslRlVecEnvWrapper(env)
                if controller == "policy":
                    runner = MjlabOnPolicyRunner(
                        wrapped, read_json(run / "agent.json"), device=device
                    )
                    runner.load(
                        str(checkpoint),
                        load_cfg={"actor": True},
                        strict=True,
                        map_location=device,
                    )
                    policy = runner.get_inference_policy(device=device)
                cfg.recorders = {
                    "trace": RecorderTermCfg(
                        func=RolloutRecorder,
                        params={"directory": str(directory), "every": 1, "worlds": 1},
                    )
                }
                env.recorder_manager = RecorderManager(cfg.recorders, env)
                env.reset(seed=seed)
                export_scene(env, directory / "scene")
                fields = task_fields(env, 0)
                initial_states.append(
                    np.concatenate(
                        (
                            env.sim.data.qpos[0].cpu().numpy(),
                            np.ravel(fields["targets"]),
                            np.ravel(fields["command"]),
                        )
                    )
                )
                if controller == "scripted":
                    scripted = make_controller(env)
                phases = []
                with torch.inference_mode():
                    for _ in range(env.max_episode_length):
                        if control is not None:
                            control()
                        if controller == "scripted":
                            phases.append(scripted.phase)
                            action = scripted()
                        else:
                            phases.append("Policy")
                            action = policy(wrapped.get_observations())
                        _, _, terminated, timeout, _ = env.step(action)
                        if bool(terminated[0] | timeout[0]):
                            break
                record = read_jsonl(directory / "workers/0/episodes.jsonl")[-1]
                with np.load(directory / "workers/0" / record["trace"]) as archive:
                    trace = dict(archive)
                report = {
                    **record,
                    **diagnostics(trace, manifest["stage"]),
                    "seed": seed,
                    "controller": controller,
                    "directory": str(directory.relative_to(output)),
                    "phases": phases,
                }
                report["events"].append(
                    {
                        "kind": "success"
                        if record["is_success"]
                        else "timeout"
                        if record["truncated"]
                        else "terminated",
                        "time": float(trace["time"][-1]),
                    }
                )
                report["return_per_second"] = record["return"] / max(
                    float(trace["time"][-1]), 1e-9
                )
                reports.append(report)
            finally:
                env.close()
        np.testing.assert_array_equal(*initial_states)
    write_json(
        output / "comparison.json",
        {
            "schema_version": 1,
            "backend": "mjlab",
            "stage": manifest["stage"],
            "training_run": str(run.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "same_initial_state_verified": True,
            "reward_terms": manifest["reward_terms"],
            "reports": reports,
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000003])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = list(args.run.glob("model_*.pt"))
        if not candidates:
            parser.error("No model_*.pt checkpoint in run")
        checkpoint = max(candidates, key=lambda path: int(path.stem.split("_")[-1]))
    output = new_run(args.output_root, "mjlab-compare")
    print(evaluate(args.run, checkpoint, output, args.seeds, args.device))


if __name__ == "__main__":
    main()
