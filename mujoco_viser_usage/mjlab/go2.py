"""Go2 residual-trot velocity task on MuJoCo-Warp worlds.

Mirrors the SB3 ``go2_velocity`` environment: the policy adds a bounded residual to
the scripted trot, and every world uses the same bundled maze/terrain geometry.
"""

import math
from dataclasses import dataclass
from functools import partial

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import nan_detection, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from ..scenes import ROOT, TERRAINS
from .assets import copy_world_geoms

STAGE = "go2"
FEET = ("FL", "FR", "RL", "RR")
LEG_JOINTS = ("hip", "thigh", "calf")
HOME = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
GOAL_DISTANCE = 3.0
LEG_LENGTH = 0.213


def terrain_xml(terrain="flat"):
    if terrain not in TERRAINS:
        raise ValueError(f"Unknown terrain: {terrain}")
    name = "go2_maze" if terrain == "flat" else terrain
    return ROOT / "locomotion/go2" / f"{name}.xml"


def robot_spec():
    directory = ROOT / "locomotion/go2"
    assets = {
        path.name: path.read_bytes() for path in (directory / "assets").glob("*.obj")
    }
    spec = mujoco.MjSpec.from_file(str(directory / "go2.xml"), assets=assets)
    for key in list(spec.keys):
        spec.delete(key)
    # mjlab adds position servos with the trot gains and the XML torque limits.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    # MuJoCo-Warp rejects margins on box pairs under MULTICCD; the XML uses 1 mm.
    for geom in spec.geoms:
        geom.margin = 0.0
    return spec


def world(spec, terrain):
    source = mujoco.MjModel.from_xml_path(str(terrain_xml(terrain)))
    copy_world_geoms(spec, source, prefix="terrain")
    spec.worldbody.add_light(
        pos=(0, 0, 8), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )


def scene_config(num_envs=1, terrain="flat"):
    servo = {"stiffness": 150.0, "damping": 4.0}
    robot = EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            # Keyframe height 0.27 starts the feet 22 mm inside the floor; stand clear.
            pos=(0.0, 0.0, 0.292),
            joint_pos={f".*_{part}_joint": value for part, value in HOME.items()},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                    effort_limit=23.7,
                    **servo,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_calf_joint",), effort_limit=45.43, **servo
                ),
            )
        ),
    )
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=0,
        entities={"robot": robot},
        sensors=(
            ContactSensorCfg(
                name="feet_contact",
                primary=ContactMatch(mode="geom", pattern=FEET, entity="robot"),
                fields=("found", "force", "dist", "pos", "normal", "tangent"),
                reduce="maxforce",
                global_frame=True,
            ),
        ),
        spec_fn=partial(world, terrain=terrain),
        extent=14.0,
    )


class Go2State:
    def __init__(self, env):
        model = env.sim.mj_model
        device = env.device
        self.base_id = model.body("robot/base").id
        joints = model.actuator_trnid[:, 0]
        self.qadr = torch.tensor(model.jnt_qposadr[joints], device=device)
        self.dofs = torch.tensor(model.jnt_dofadr[joints], device=device)
        self.limits = torch.tensor(
            model.jnt_range[joints], device=device, dtype=torch.float32
        )
        # Servo order follows the actuator configs, so index the analytic gait by name.
        gait_index = []
        home = []
        for joint in joints:
            leg, part = model.joint(int(joint)).name.split("/")[-1].split("_")[:2]
            gait_index.append(FEET.index(leg) * 3 + LEG_JOINTS.index(part))
            home.append(HOME[part])
        self.gait_index = torch.tensor(gait_index, device=device)
        self.home = torch.tensor(home, device=device, dtype=torch.float32)
        self.command = torch.full((env.num_envs,), 0.3, device=device)
        self.start_x = torch.zeros(env.num_envs, device=device)


def state(env):
    if not hasattr(env, "_go2_state"):
        env._go2_state = Go2State(env)
    return env._go2_state


def gait_targets(time, stride, frequency=2.0, lift=0.08, height=0.28):
    """Tensorized two-link leg IK trot; returns FL/FR/RL/RR × hip/thigh/calf."""
    offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=time.device)
    phase = (time[:, None] * frequency + offsets) % 1.0
    stance = phase < 0.6
    swing = ((phase - 0.6) / 0.4).clamp(0, 1)
    stride = stride[:, None]
    x = torch.where(stance, stride * (0.5 - phase / 0.6), stride * (swing - 0.5))
    z = torch.where(
        stance,
        torch.full_like(phase, -height),
        -height + lift * torch.sin(math.pi * swing),
    )
    cosine = (x * x + z * z - 2 * LEG_LENGTH**2) / (2 * LEG_LENGTH**2)
    knee = -torch.acos(cosine.clamp(-1.0, 1.0))
    thigh = torch.atan2(-x, -z) - knee / 2
    return torch.stack((torch.zeros_like(thigh), thigh, knee), dim=-1).flatten(1)


def elapsed(env):
    return env.episode_length_buf.float() * env.step_dt


def base_position(env):
    return env.sim.data.xpos[:, state(env).base_id]


def robot(env):
    return env.scene["robot"]


def upright(env):
    return -robot(env).data.projected_gravity_b[:, 2]


def reset(env, env_ids, stage):
    del stage
    reset_scene_to_default(env, env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cache = state(env)
    noise = torch.rand((len(env_ids), len(cache.qadr)), device=env.device) - 0.5
    env.sim.data.qpos[env_ids[:, None], cache.qadr[None, :]] += 0.03 * noise
    cache.command[env_ids] = 0.2 + 0.25 * torch.rand(len(env_ids), device=env.device)
    cache.start_x[env_ids] = (
        robot(env).data.default_root_state[env_ids, 0]
        + env.scene.env_origins[env_ids, 0]
    )


def observation(env):
    cache = state(env)
    data = robot(env).data
    phase = 2 * math.pi * 2 * elapsed(env)
    return torch.cat(
        (
            env.sim.data.qpos[:, cache.qadr],
            0.1 * env.sim.data.qvel[:, cache.dofs],
            env.sim.data.ctrl[:],
            env.action_manager.action,
            data.projected_gravity_b,
            data.root_link_lin_vel_b,
            data.root_link_ang_vel_b,
            torch.stack(
                (base_position(env)[:, 2], cache.command, phase.sin(), phase.cos()),
                dim=-1,
            ),
        ),
        dim=-1,
    )


def velocity_reward(env):
    forward = robot(env).data.root_link_lin_vel_b[:, 0]
    return torch.exp(-(((forward - state(env).command) / 0.25) ** 2))


def upright_reward(env):
    return upright(env)


def lateral_cost(env):
    return robot(env).data.root_link_lin_vel_b[:, 1] ** 2


def angular_cost(env):
    return robot(env).data.root_link_ang_vel_b.square().sum(-1)


def action_cost(env):
    return env.action_manager.action.square().mean(-1)


def smoothness_cost(env):
    return (
        (env.action_manager.action - env.action_manager.prev_action).square().mean(-1)
    )


def fallen(env):
    base = base_position(env)
    return (base[:, 2] < 0.16) | (upright(env) < 0.6) | (base[:, 1].abs() > 0.65)


def fall_cost(env):
    return fallen(env).float()


def success(env):
    return base_position(env)[:, 0] - state(env).start_x > GOAL_DISTANCE


@dataclass(kw_only=True)
class Go2ActionCfg(ActionTermCfg):
    def build(self, env):
        return Go2Action(self, env)


class Go2Action(ActionTerm):
    """Residual joint targets over the analytic trot, through position servos."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.sim.mj_model
        self._raw = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        global_ids = self._entity.indexing.joint_ids.cpu().tolist()
        self.joint_ids = torch.tensor(
            [global_ids.index(int(j)) for j in model.actuator_trnid[:, 0]],
            device=env.device,
        )
        self.targets = state(env).home.repeat(env.num_envs, 1)

    @property
    def action_dim(self):
        return 12

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1, 1)

    def nominal(self):
        cache = state(self._env)
        time = elapsed(self._env)
        ramp = time.clamp(0, 1)[:, None]
        gait = gait_targets(time, 0.2 * cache.command / 0.4)[:, cache.gait_index]
        return cache.home * (1 - ramp) + gait * ramp

    def apply_actions(self):
        cache = state(self._env)
        self.targets[:] = (self.nominal() + 0.15 * self._raw).clamp(
            cache.limits[:, 0], cache.limits[:, 1]
        )
        self._entity.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.targets[ids] = state(self._env).home
        self._entity.set_joint_position_target(
            self.targets[ids], joint_ids=self.joint_ids, env_ids=env_ids
        )


def environment_config(num_envs=1, seed=7, terrain="flat"):
    observations = {
        name: ObservationGroupCfg(
            terms={"state": ObservationTermCfg(func=observation)},
            enable_corruption=False,
        )
        for name in ("actor", "critic")
    }
    return ManagerBasedRlEnvCfg(
        scene=scene_config(num_envs, terrain),
        seed=seed,
        decimation=10,
        episode_length_s=10,
        observations=observations,
        actions={"servo": Go2ActionCfg(entity_name="robot")},
        events={
            "reset": EventTermCfg(func=reset, mode="reset", params={"stage": STAGE})
        },
        rewards={
            "velocity": RewardTermCfg(func=velocity_reward, weight=1),
            "upright": RewardTermCfg(func=upright_reward, weight=0.2),
            "lateral": RewardTermCfg(func=lateral_cost, weight=-0.2),
            "angular": RewardTermCfg(func=angular_cost, weight=-0.02),
            "fall": RewardTermCfg(func=fall_cost, weight=-3),
            "action_cost": RewardTermCfg(func=action_cost, weight=-0.005),
            "smoothness": RewardTermCfg(func=smoothness_cost, weight=-0.005),
        },
        terminations={
            "success": TerminationTermCfg(func=success),
            "fall": TerminationTermCfg(func=fallen),
            "nan": TerminationTermCfg(func=nan_detection),
            "time_limit": TerminationTermCfg(func=time_out, time_out=True),
        },
        viewer=ViewerConfig(
            lookat=(1.5, 0, 0.2), distance=4, elevation=-25, azimuth=135
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=2048,
            mujoco=MujocoCfg(
                timestep=0.002,
                integrator="implicitfast",
                cone="elliptic",
                impratio=100,
                iterations=30,
                ls_iterations=20,
            ),
        ),
    )


class ScriptedTrot:
    """Zero residual: the analytic trot alone, through the same action interface."""

    phase = "Trot"

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("Scripted evaluation requires one world")
        self.env = env

    def __call__(self):
        return torch.zeros((1, 12), device=self.env.device)


def snapshot(env, world):
    cache = state(env)
    contact = env.scene["feet_contact"].data
    frame = {
        "targets": np.zeros((2, 3)),
        "command": cache.command[world].cpu().numpy().copy(),
        "base_pos": base_position(env)[world].cpu().numpy().copy(),
        "base_upright": upright(env)[world].cpu().numpy().copy(),
    }
    for field in ("found", "force", "dist", "pos"):
        frame[f"contact_{field}"] = (
            getattr(contact, field)[world].detach().cpu().numpy().copy()
        )
    return frame


def diagnostics(trace):
    base = trace["base_pos"]
    traveled = base[:, 0] - base[0, 0]
    force = np.linalg.norm(trace["contact_force"], axis=-1).reshape(len(base), -1)
    penetration = np.maximum(0, -trace["contact_dist"].reshape(len(base), -1).min(-1))
    fallen = (base[:, 2] < 0.16) | (trace["base_upright"] < 0.6)
    events = []
    for kind, mask in [("fall", fallen), ("penetration", penetration > 0.003)]:
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        events.extend({"kind": kind, "time": float(trace["time"][i])} for i in starts)
    return {
        "final_goal_error_m": float(max(0.0, GOAL_DISTANCE - traveled[-1])),
        "distance_traveled_m": float(traveled[-1]),
        "min_base_height_m": float(base[:, 2].min()),
        "max_recorded_contact_force_n": float(force.max()),
        "max_recorded_penetration_m": float(penetration.max()),
        "events": sorted(events, key=lambda event: event["time"]),
    }
