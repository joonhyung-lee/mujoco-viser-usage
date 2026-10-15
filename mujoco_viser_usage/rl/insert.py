"""Franka insertion on the NIST board: 25 Hz task-space deltas plus a gripper command.

Episodes start from a task configuration (see configs.py): the part seated,
hovering, or lying somewhere on the table. The hole offset is never observed;
the policy sees proprioception, the part pose relative to the TCP (what a
point cloud would give), and the wrench on the part.
"""

from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..kinematics import SiteIK
from ..scenes.franka_task import CLOSED, JOINTS, OPEN, FrankaTask, grasp_rotation
from ..simulation import Simulation
from .config import TrainConfig, scene_path
from .configs import (
    WORKSPACE_X,
    WORKSPACE_Y,
    InsertConfig,
    sample_config,
    sample_configs,
)

GRASP_OFFSET = 0.012  # TCP above the part center, as in the scripted NIST run
START_HEIGHT = 0.03  # part center above its seated pose at reset
ACTION_SCALE = np.array([0.004, 0.004, 0.004, np.radians(4)])  # per 25 Hz step
SUCCESS = 0.001  # AlphaNIST: within 1 mm of the assembled pose
# Sparse reward from the SGS paper: success, failure, action, action rate, joint speed.
SPARSE = {"success": 100.0, "fail": 1.0, "action": 1e-5, "rate": 1e-4, "speed": 1e-4}


class FrankaInsertEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: TrainConfig):
        self.config = config
        self.part = config.part
        self.sim = Simulation(scene_path(config))
        self.model, self.data = self.sim.model, self.sim.data
        self.frame_skip = round(config.control_dt / self.model.opt.timestep)
        if not np.isclose(self.frame_skip * self.model.opt.timestep, config.control_dt):
            raise ValueError("control_dt must be a multiple of the physics timestep")
        FrankaTask(self.sim, "nist")  # stiffens the gripper; the task itself is unused
        self.ik = SiteIK(self.model, "tcp", JOINTS)
        self.arm_dofs = self.ik.dofs
        # The real Panda compensates gravity itself; without it the position
        # servos sag 5-7 mm at the TCP, which is the whole insertion budget.
        self.sim.before_step = self._gravity_compensation
        self.fingers = self.model.jnt_qposadr[
            [self.model.joint(f"finger_joint{i}").id for i in (1, 2)]
        ]
        self.body = self.model.body(self.part).id
        self.part_qadr = self.model.jnt_qposadr[self.model.body(self.part).jntadr[0]]
        self.goal = self.data.site(f"{self.part}_goal").xpos.copy()
        geom = self.model.body(self.part).geomadr[0]
        # Pads close along the gripper y axis: box half-width y, or the rod radius.
        self.half_width = float(
            self.model.geom_size[
                geom,
                1 if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX else 0,
            ]
        )
        self.half_length = float(
            self.model.geom_size[
                geom,
                2 if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX else 1,
            ]
        )
        self.configs = (
            sample_configs(config.configs, self.part, config.seed)
            if config.configs
            else []
        )
        self.assigned = None  # index into self.configs, set by the sampler
        self.config_index = -1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.target = np.zeros(4)  # TCP x, y, z, yaw
        self.previous_action = np.zeros(5)
        self.elapsed_steps = 0
        self.targets = self.goal[None]  # EpisodeRecorder snapshots these
        self.command = 0.0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self._observation().shape, dtype=np.float32
        )
        self._needs_reset = True

    def assign(self, index):
        """Choose the configuration for the next reset (called by the sampler)."""
        self.assigned = int(index)

    def _gravity_compensation(self):
        self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]

    def _rotation(self, yaw):
        return grasp_rotation((-np.sin(yaw), -np.cos(yaw), 0.0))

    def _part_position(self):
        return self.data.body(self.part).xpos

    def _observation(self):
        tcp = self.data.site("tcp").xpos
        part = self._part_position()
        state = [
            self.data.qpos[self.ik.qadr],
            0.1 * self.data.qvel[self.arm_dofs],
            10 * (tcp - self.goal),
            10 * (self.target[:3] - self.goal),
            [self.target[3]],
            10 * (part - tcp),
            self.data.xmat[self.body].reshape(3, 3)[:, 2],
            [self.data.qpos[self.fingers].sum() * 10],
            0.1 * self.data.cfrc_ext[self.body],
            self.previous_action,
        ]
        return np.concatenate(state).astype(np.float32)

    def _drive(self):
        """IK the TCP target into arm ctrl; keep the previous ctrl if IK fails."""
        try:
            q = self.ik.solve(
                self.data.qpos, self.target[:3], self._rotation(self.target[3])
            )
        except ValueError:
            return
        self.data.ctrl[: len(q)] = q

    def _solve_start(self, yaw):
        """IK for the start pose. The grasp is symmetric under a half turn, and
        joint 7 near its limit traps the solver, so try both yaws from a few seeds."""
        home = self.model.key_qpos[0]
        for turn in (0.0, np.pi):
            for j7 in (0.0, 1.5, -1.5):
                seed = home.copy()
                seed[self.ik.qadr[6]] = np.clip(
                    seed[self.ik.qadr[6]] + j7, *self.model.jnt_range[self.ik.joints[6]]
                )
                try:
                    q = self.ik.solve(seed, self.target[:3], self._rotation(yaw + turn))
                except ValueError:
                    continue
                self.target[3] = yaw + turn
                return q
        raise ValueError(f"No start pose for {self.target[:3].tolist()}")

    def _place(self, cfg: InsertConfig):
        """Put the part and the gripper where the configuration says, then settle."""
        if cfg.strategy == "reaching":
            # Standing on the table or plate; the plate top is 9 mm above the table.
            on_plate = abs(cfg.place_x - 0.52) < 0.192 and abs(cfg.place_y) < 0.192
            part = np.array(
                [
                    cfg.place_x,
                    cfg.place_y,
                    (0.009 if on_plate else 0) + self.half_length + 0.001,
                ]
            )
            width = self.half_width + 0.0005 + cfg.aperture * (0.04 - self.half_width)
            gripper = OPEN if cfg.aperture > 0.5 else CLOSED
        else:
            height = START_HEIGHT * (
                cfg.fraction if cfg.strategy == "near_goal" else 1.0
            )
            # The part is inside the hole below 9 mm; the hole offset only applies above it.
            offset = np.clip((height - 0.009) / 0.003, 0, 1)
            part = self.goal + [cfg.hole_x * offset, cfg.hole_y * offset, height]
            width = self.half_width + 0.0005
            gripper = CLOSED
        self.target[:] = [
            part[0] + cfg.in_hand_x,
            part[1],
            part[2] + GRASP_OFFSET,
            cfg.yaw,
        ]
        q = self._solve_start(cfg.yaw)
        self.data.qpos[self.ik.qadr] = q
        self.data.qpos[self.fingers] = width
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), cfg.yaw + cfg.twist)
        self.data.qpos[self.part_qadr : self.part_qadr + 7] = [*part, *quat]
        self.data.qvel[:] = 0
        self.data.ctrl[: len(q)] = q
        self.data.ctrl[len(q)] = gripper
        self.gripper_closed = gripper == CLOSED
        # Pads close on the part without gravity so the grasp is where we placed it.
        gravity = self.model.opt.gravity.copy()
        self.model.opt.gravity[:] = 0
        self.sim.step(round(0.3 / self.model.opt.timestep))
        self.model.opt.gravity[:] = gravity

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.elapsed_steps = 0
        self.previous_action[:] = 0
        self._needs_reset = False
        cfg = (options or {}).get("config")
        if cfg is None and self.assigned is not None:
            self.config_index, cfg = self.assigned, self.configs[self.assigned]
        elif cfg is None:
            # No buffer: a fresh hovering start each episode (the v1 benchmark).
            self.config_index, cfg = (
                -1,
                sample_config(self.np_random, self.part, "stable"),
            )
        else:
            self.config_index = -1
        self.current = cfg
        self._place(cfg)
        self.previous_distance = self._distance()
        return self._observation(), self._info(False, {})

    def _distance(self):
        return float(np.linalg.norm(self._part_position() - self.goal))

    def _info(self, success, terms):
        return {
            "is_success": bool(success),
            "reward_terms": terms,
            "distance": self._distance(),
            "sim_time": float(self.data.time),
            "control_steps": self.elapsed_steps,
            "max_penetration": max(0.0, -float(min(self.data.contact.dist, default=0))),
            "config": self.config_index,
            "strategy": self.current.strategy,
        }

    def scripted_action(self):
        """Straight down to the nominal goal with the gripper closed; the hovering baseline."""
        seat = np.array([*self.goal, 0.0]) + [0, 0, GRASP_OFFSET, 0]
        return np.array([*np.clip((seat - self.target) / ACTION_SCALE, -1, 1), 1.0])

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset() is required before stepping a new episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Action must contain five finite values")
        action = np.clip(action, -1, 1)
        self.target += ACTION_SCALE * action[:4]
        low = [
            WORKSPACE_X[0],
            WORKSPACE_Y[0],
            self.goal[2] + GRASP_OFFSET - 0.002,
            -0.6,
        ]
        high = [WORKSPACE_X[1], WORKSPACE_Y[1], 0.25, 0.6]
        self.target = np.clip(self.target, low, high)
        self.gripper_closed = action[4] > 0
        self.data.ctrl[7] = CLOSED if self.gripper_closed else OPEN
        self._drive()
        self.sim.step(self.frame_skip)
        self.elapsed_steps += 1
        finite = np.isfinite(self.data.qpos).all()
        distance = self._distance()
        part = self._part_position()
        success = distance < SUCCESS
        # Failure as in the paper: the part left the workspace (fell off the table).
        failed = bool(
            part[2] < -0.05
            or not (WORKSPACE_X[0] - 0.1 < part[0] < WORKSPACE_X[1] + 0.1)
            or not (WORKSPACE_Y[0] - 0.1 < part[1] < WORKSPACE_Y[1] + 0.1)
        )
        if self.config.reward == "sparse":
            terms = {
                "success": SPARSE["success"] * float(success),
                "fail": -SPARSE["fail"] * float(failed),
                "action": -SPARSE["action"] * float(np.sum(action**2)),
                "rate": -SPARSE["rate"]
                * float(np.sum((action - self.previous_action) ** 2)),
                "speed": -SPARSE["speed"]
                * float(min(np.sum(self.data.qvel[self.arm_dofs] ** 2), 1e4)),
            }
        else:
            terms = {
                "progress": 50.0 * (self.previous_distance - distance),
                "proximity": 0.1 * float(np.exp(-distance / 0.01)),
                "success": 10.0 * float(success),
                "fail": -2.0 * float(failed),
                "action_cost": -0.005 * float(np.mean(action**2)),
                "smoothness": -0.005
                * float(np.mean((action - self.previous_action) ** 2)),
            }
        self.previous_distance = distance
        self.previous_action = action.copy()
        terminated = bool(success or failed or not finite)
        truncated = bool(
            not terminated and self.elapsed_steps >= self.config.max_episode_steps
        )
        self._needs_reset = terminated or truncated
        observation = np.nan_to_num(
            self._observation(), nan=0.0, posinf=100.0, neginf=-100.0
        )
        reward = float(sum(terms.values())) if finite else -10.0
        return observation, reward, terminated, truncated, self._info(success, terms)
