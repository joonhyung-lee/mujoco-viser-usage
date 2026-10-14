"""Tensorized ALOHA tasks, actions and diagnostics for MuJoCo-Warp worlds."""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


class AlohaState:
    def __init__(self, env):
        self.stage = env.cfg.events["reset"].params["stage"]
        model = env.sim.mj_model
        self.site_ids = [
            model.site("robot/" + side + "/gripper").id for side in ("left", "right")
        ]
        self.body_ids = [model.body(f"object_{i}/object_{i}").id for i in range(2)]
        self.qadr = torch.tensor(
            model.jnt_qposadr[model.actuator_trnid[:, 0]], device=env.device
        )
        self.dofs = torch.tensor(
            model.jnt_dofadr[model.actuator_trnid[:, 0]], device=env.device
        )
        self.goals = torch.zeros((env.num_envs, 2, 3), device=env.device)
        self.lifted = torch.zeros(
            (env.num_envs, 2), dtype=torch.bool, device=env.device
        )
        self.held_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.reach_goals = None
        if self.stage == "reach":
            # A shared reachable goal bank avoids per-world CPU FK at every reset.
            scratch = mujoco.MjData(model)
            arm_ids = [
                i for i in range(model.nu) if "gripper" not in model.actuator(i).name
            ]
            qadr = model.jnt_qposadr[model.actuator_trnid[arm_ids, 0]]
            rng = np.random.default_rng(0)
            goals = []
            for delta in rng.uniform(-0.2, 0.2, (1024, len(arm_ids))):
                scratch.qpos[:] = model.key_qpos[0]
                scratch.qpos[qadr] += delta
                mujoco.mj_forward(model, scratch)
                goals.append(scratch.site_xpos[self.site_ids].copy())
            self.reach_goals = torch.tensor(
                np.asarray(goals), device=env.device, dtype=torch.float32
            )


def state(env):
    if not hasattr(env, "_aloha_state"):
        env._aloha_state = AlohaState(env)
    return env._aloha_state


def tools(env):
    return env.sim.data.site_xpos[:, state(env).site_ids]


def objects(env):
    return env.sim.data.xpos[:, state(env).body_ids]


def reset(env, env_ids, stage):
    reset_scene_to_default(env, env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cache = state(env)
    cache.lifted[env_ids] = False
    cache.held_steps[env_ids] = 0
    for i in range(2):
        obj = env.scene[f"object_{i}"]
        pose = obj.data.default_root_state[env_ids].clone()
        pose[:, :2] += (torch.rand((len(env_ids), 2), device=env.device) - 0.5) * 0.03
        obj.write_root_state_to_sim(pose, env_ids=env_ids)
    if stage == "reach":
        indices = torch.randint(
            len(cache.reach_goals), (len(env_ids),), device=env.device
        )
        cache.goals[env_ids] = cache.reach_goals[indices]
    else:
        cache.goals[env_ids] = torch.tensor(
            [[-0.14, 0.23, 0.03], [0.14, 0.23, 0.03]], device=env.device
        )
        if stage == "lift":
            cache.goals[env_ids, 0] = torch.tensor([0, -0.12, 0.15], device=env.device)


def observation(env):
    cache = state(env)
    return torch.cat(
        (
            env.sim.data.qpos[:, cache.qadr],
            0.1 * env.sim.data.qvel[:, cache.dofs],
            env.sim.data.ctrl[:],
            env.action_manager.action,
            tools(env).flatten(1),
            objects(env).flatten(1),
            cache.goals.flatten(1),
            (cache.goals - tools(env)).flatten(1),
        ),
        dim=-1,
    )


def reach_reward(env):
    distance = torch.linalg.vector_norm(tools(env) - state(env).goals, dim=-1)
    return torch.exp(-10 * distance).mean(-1)


def approach_reward(env):
    distance = torch.linalg.vector_norm(tools(env) - objects(env), dim=-1)
    return torch.exp(-10 * distance[:, : active_count(env)]).mean(-1)


def active_count(env):
    return 2 if state(env).stage in {"reach", "bimanual"} else 1


def lift_reward(env):
    return torch.clamp(
        (objects(env)[:, : active_count(env), 2] - 0.03) / 0.12, 0, 1
    ).mean(-1)


def place_reward(env):
    cache = state(env)
    distance = torch.linalg.vector_norm(objects(env) - cache.goals, dim=-1)
    # Avoid rewarding pushing as successful pick/place: each object must first lift.
    return (torch.exp(-12 * distance) * cache.lifted)[:, : active_count(env)].mean(-1)


def update_progress(env, env_ids=None):
    del env_ids
    cache = state(env)
    cache.lifted |= objects(env)[:, :, 2] > 0.10
    if cache.stage == "reach":
        reached = torch.linalg.vector_norm(tools(env) - cache.goals, dim=-1) < 0.025
    elif cache.stage == "lift":
        reached = objects(env)[:, :, 2] > 0.12
    else:
        reached = (
            torch.linalg.vector_norm(objects(env) - cache.goals, dim=-1) < 0.025
        ) & cache.lifted
        # Release required for place; finger target normalized by actuator range.
        reached &= env.sim.data.ctrl[:, [6, 13]] > 0.025
    ok = reached[:, : active_count(env)].all(-1)
    cache.held_steps[:] = torch.where(ok, cache.held_steps + 1, 0)


def success(env):
    update_progress(env)
    return state(env).held_steps >= 10


def object_dropped(env):
    return (objects(env)[:, : active_count(env), 2] < -0.08).any(-1)


def action_cost(env):
    return env.action_manager.action.square().mean(-1)


def smoothness_cost(env):
    return (
        (env.action_manager.action - env.action_manager.prev_action).square().mean(-1)
    )


@dataclass(kw_only=True)
class AlohaActionCfg(ActionTermCfg):
    single_arm: bool = False
    incremental: bool = False

    def build(self, env):
        return AlohaAction(self, env)


class AlohaAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.sim.mj_model
        self._raw = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        self.home = torch.tensor(
            model.key_ctrl[0], device=env.device, dtype=torch.float32
        )
        self.targets = self.home.repeat(env.num_envs, 1)
        self.limits = torch.tensor(
            model.actuator_ctrlrange, device=env.device, dtype=torch.float32
        )
        # Leaf joint names repeat across arms; map by compiled IDs instead.
        global_ids = self._entity.indexing.joint_ids.cpu().tolist()
        self.joint_ids = torch.tensor(
            [global_ids.index(int(j)) for j in model.actuator_trnid[:, 0]],
            device=env.device,
        )
        self.arm_dofs = torch.tensor(
            [
                model.jnt_dofadr[j]
                for i, j in enumerate(model.actuator_trnid[:, 0])
                if i not in (6, 13)
            ],
            device=env.device,
        )

    @property
    def action_dim(self):
        return 7 if self.cfg.single_arm else 14

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1, 1)
        if not self.cfg.incremental:
            self.targets[:] = self.home
        scale = 0.04 if self.cfg.incremental else 0.6
        self.targets[:, : self.action_dim] += scale * self._raw
        for i in (6, 13):
            if i < self.action_dim:
                self.targets[:, i] = self.limits[i, 0] + (
                    0.5 * self._raw[:, i] + 0.5
                ) * (self.limits[i, 1] - self.limits[i, 0])
        self.targets[:] = self.targets.clamp(self.limits[:, 0], self.limits[:, 1])

    def apply_actions(self):
        self._entity.set_joint_position_target(self.targets, joint_ids=self.joint_ids)
        self._env.sim.data.qfrc_applied[:, self.arm_dofs] = (
            self._env.sim.data.qfrc_bias[:, self.arm_dofs]
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.targets[ids] = self.home
        self._entity.set_joint_position_target(
            self.targets[ids], joint_ids=self.joint_ids, env_ids=env_ids
        )


def reach_cost(env):
    """Nonpositive shaping avoids rewarding a policy for postponing success.

    Keep reach_reward unchanged for configurations saved by earlier runs.
    """
    return reach_reward(env) - 1.0


def completion_reward(env):
    """A fixed episode bonus, independent of the control timestep."""
    return env.termination_manager.get_term("success").float() / env.step_dt
