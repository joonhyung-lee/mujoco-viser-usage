"""Four ALOHA stages registered as native mjlab manager-based environments."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import nan_detection, time_out
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from . import TASK_IDS, mdp
from .assets import scene_config


def environment_config(stage="reach", num_envs=1, seed=7):
    if stage not in TASK_IDS:
        raise ValueError(f"Unknown ALOHA stage: {stage}")
    rewards = {
        "action_cost": RewardTermCfg(func=mdp.action_cost, weight=-0.005),
        "smoothness": RewardTermCfg(func=mdp.smoothness_cost, weight=-0.01),
    }
    if stage == "reach":
        rewards["reach"] = RewardTermCfg(func=mdp.reach_cost, weight=1)
        rewards["success"] = RewardTermCfg(func=mdp.completion_reward, weight=2)
    else:
        rewards["approach"] = RewardTermCfg(func=mdp.approach_reward, weight=0.5)
        rewards["lift"] = RewardTermCfg(func=mdp.lift_reward, weight=1)
        if stage != "lift":
            rewards["place"] = RewardTermCfg(func=mdp.place_reward, weight=2)
    observations = {
        name: ObservationGroupCfg(
            terms={"state": ObservationTermCfg(func=mdp.observation)},
            enable_corruption=False,
        )
        for name in ("actor", "critic")
    }
    return ManagerBasedRlEnvCfg(
        scene=scene_config(num_envs),
        seed=seed,
        decimation=10,
        episode_length_s={"reach": 8, "lift": 12, "place": 22, "bimanual": 42}[stage],
        observations=observations,
        actions={
            "servo": mdp.AlohaActionCfg(
                entity_name="robot",
                single_arm=stage in {"lift", "place"},
                incremental=stage != "reach",
            )
        },
        events={
            "reset": EventTermCfg(func=mdp.reset, mode="reset", params={"stage": stage})
        },
        rewards=rewards,
        terminations={
            "success": TerminationTermCfg(func=mdp.success),
            "object_drop": TerminationTermCfg(func=mdp.object_dropped),
            "nan": TerminationTermCfg(func=nan_detection),
            "time_limit": TerminationTermCfg(func=time_out, time_out=True),
        },
        viewer=ViewerConfig(
            lookat=(0, 0.04, 0.15), distance=2, elevation=-35, azimuth=130
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=1024,
            mujoco=MujocoCfg(
                timestep=0.002,
                integrator="implicitfast",
                cone="elliptic",
                impratio=10,
                iterations=30,
                ls_iterations=20,
            ),
        ),
    )


def runner_config():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.3,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(128, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5, num_mini_batches=4, learning_rate=0.0003
        ),
        logger="tensorboard",
        experiment_name="viser_aloha",
        num_steps_per_env=24,
        max_iterations=1000,
        save_interval=50,
    )


def register_tasks():
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    from .runner import RecordedRunner

    for stage, task_id in TASK_IDS.items():
        if task_id not in list_tasks():
            register_mjlab_task(
                task_id,
                environment_config(stage),
                environment_config(stage),
                runner_config(),
                runner_cls=RecordedRunner,
            )
