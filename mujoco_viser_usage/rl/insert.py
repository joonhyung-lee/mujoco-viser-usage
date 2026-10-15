"""Franka peg insertion on the NIST board: part already grasped, 25 Hz task-space deltas."""

from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..kinematics import SiteIK
from ..scenes.franka_task import CLOSED, JOINTS, FrankaTask, grasp_rotation
from ..simulation import Simulation
from .config import TrainConfig, scene_path

GRASP_OFFSET = 0.012  # TCP above the part center, as in the scripted NIST run
START_HEIGHT = 0.03  # part center above its seated pose at reset
ACTION_SCALE = np.array([0.002, 0.002, 0.003, np.radians(3)])  # per 25 Hz step
SUCCESS = 0.001  # AlphaNIST: within 1 mm of the assembled pose


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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.target = np.zeros(4)  # TCP x, y, z, yaw
        self.previous_action = np.zeros(4)
        self.elapsed_steps = 0
        self.targets = self.goal[None]  # EpisodeRecorder snapshots these
        self.command = 0.0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self._observation().shape, dtype=np.float32
        )
        self._needs_reset = True

    def _gravity_compensation(self):
        self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]

    def _rotation(self, yaw):
        return grasp_rotation((-np.sin(yaw), -np.cos(yaw), 0.0))

    def _part_position(self):
        return self.data.body(self.part).xpos

    def _observation(self):
        """Proprioception, the nominal goal, and the wrench on the part: no part pose."""
        tcp = self.data.site("tcp").xpos
        state = [
            self.data.qpos[self.ik.qadr],
            0.1 * self.data.qvel[self.arm_dofs],
            10 * (tcp - self.goal),
            10 * (self.target[:3] - self.goal),
            [self.target[3]],
            [self.data.qpos[self.fingers].sum() * 10],
            0.1 * self.data.cfrc_ext[self.model.body(self.part).id],
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.elapsed_steps = 0
        self.previous_action[:] = 0
        self._needs_reset = False
        rng = self.np_random
        # Hole offset the policy cannot see directly, plus an in-hand offset and
        # (for square pegs) an in-hand yaw. Only forces reveal them.
        part = self.goal + [*rng.uniform(-0.005, 0.005, 2), START_HEIGHT]
        in_hand = rng.uniform(-0.003, 0.003, 2)
        yaw = float(rng.uniform(-0.17, 0.17)) if self.part.startswith("peg") else 0.0
        twist = float(rng.uniform(-0.09, 0.09)) if self.part.startswith("peg") else 0.0
        self.target[:] = [*(part[:2] + in_hand), part[2] + GRASP_OFFSET, yaw]
        q = self.ik.solve(self.model.key_qpos[0], self.target[:3], self._rotation(yaw))
        self.data.qpos[self.ik.qadr] = q
        self.data.qpos[self.fingers] = self.half_width + 0.0005
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw + twist)
        self.data.qpos[self.part_qadr : self.part_qadr + 7] = [*part, *quat]
        self.data.qvel[:] = 0
        self.data.ctrl[: len(q)] = q
        self.data.ctrl[len(q)] = CLOSED
        # Pads close on the part without gravity so the grasp is where we placed it.
        gravity = self.model.opt.gravity.copy()
        self.model.opt.gravity[:] = 0
        self.sim.step(round(0.3 / self.model.opt.timestep))
        self.model.opt.gravity[:] = gravity
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
        }

    def scripted_action(self):
        """Straight down to the nominal goal; only the chamfer absorbs the offsets."""
        seat = np.array([*self.goal, 0.0]) + [0, 0, GRASP_OFFSET, 0]
        return np.clip((seat - self.target) / ACTION_SCALE, -1, 1)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset() is required before stepping a new episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Action must contain four finite values")
        action = np.clip(action, -1, 1)
        self.target += ACTION_SCALE * action
        low = [*(self.goal[:2] - 0.02), self.goal[2] + GRASP_OFFSET - 0.002, -0.6]
        high = [
            *(self.goal[:2] + 0.02),
            self.goal[2] + GRASP_OFFSET + START_HEIGHT + 0.02,
            0.6,
        ]
        self.target = np.clip(self.target, low, high)
        self._drive()
        self.sim.step(self.frame_skip)
        self.elapsed_steps += 1
        finite = np.isfinite(self.data.qpos).all()
        distance = self._distance()
        success = distance < SUCCESS
        dropped = (
            np.linalg.norm(self._part_position() - self.data.site("tcp").xpos) > 0.04
        )
        terms = {
            "progress": 50.0 * (self.previous_distance - distance),
            "proximity": 0.1 * float(np.exp(-distance / 0.01)),
            "success": 10.0 * float(success),
            "dropped": -2.0 * float(dropped),
            "action_cost": -0.005 * float(np.mean(action**2)),
            "smoothness": -0.005 * float(np.mean((action - self.previous_action) ** 2)),
        }
        self.previous_distance = distance
        self.previous_action = action.copy()
        terminated = bool(success or dropped or not finite)
        truncated = bool(
            not terminated and self.elapsed_steps >= self.config.max_episode_steps
        )
        self._needs_reset = terminated or truncated
        observation = np.nan_to_num(
            self._observation(), nan=0.0, posinf=100.0, neginf=-100.0
        )
        reward = float(sum(terms.values())) if finite else -10.0
        return observation, reward, terminated, truncated, self._info(success, terms)
