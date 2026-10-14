"""Native RSL-RL runner with reproducible scene and bounded rollout artifacts."""

import os
import pickle
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from mjlab.managers.recorder_manager import RecorderManager, RecorderTermCfg
from mjlab.rl import MjlabOnPolicyRunner

from ..rl.storage import write_json
from .assets import export_scene
from .recording import RolloutRecorder


class RecordedRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        if log_dir is not None:
            # RSL-RL mutates its input dictionary during construction.
            write_json(Path(log_dir) / "agent.json", train_cfg)
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if log_dir is None:
            return
        base = env.unwrapped
        directory = Path(log_dir)
        export_scene(base, directory / "scene")
        saved_environment = deepcopy(base.cfg)
        saved_environment.recorders = {}
        (directory / "environment.pkl").write_bytes(pickle.dumps(saved_environment))
        write_json(
            directory / "run.json",
            {
                "schema_version": 1,
                "reward_terms": list(base.cfg.rewards),
                "backend": "mjlab",
                "algorithm": "rsl_rl.PPO",
                "stage": base.cfg.events["reset"].params["stage"],
                "seed": base.cfg.seed,
                "num_envs": base.num_envs,
                "step_dt": base.step_dt,
                "episode_length_s": base.cfg.episode_length_s,
                "device": str(device),
                "scene": "scene/scene.xml",
                "versions": {
                    name: version(name)
                    for name in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib")
                },
            },
        )
        base.recorder_manager.close()
        base.cfg.recorders = {
            "rollouts": RecorderTermCfg(
                func=RolloutRecorder,
                params={"directory": str(directory), "worlds": 4, "every": 5},
            )
        }
        base.recorder_manager = RecorderManager(base.cfg.recorders, base)
        # The wrapper reset happened before runner construction. Start recording
        # from a fresh, explicitly seeded state, not halfway through an episode.
        env.reset()

    def save(self, path, infos=None):
        """Publish a whole checkpoint so live evaluation never opens a partial file."""
        destination = Path(path)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        upload = self.cfg["upload_model"]
        try:
            self.cfg["upload_model"] = False
            super().save(str(temporary), infos)
            os.replace(temporary, destination)
        finally:
            self.cfg["upload_model"] = upload
            temporary.unlink(missing_ok=True)
        if upload:
            self.logger.save_model(str(destination), self.current_learning_iteration)
