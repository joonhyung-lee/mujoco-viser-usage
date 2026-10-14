"""Gymnasium episodes over the same MuJoCo robots used in the interactive scenes."""

from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..scenes.go2_task import configure_servos, gait_targets
from ..simulation import Simulation
from .config import TASK_SCENES, TrainConfig, scene_path


class RobotEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: TrainConfig):
        self.config = config
        self.scene_id = TASK_SCENES[config.task]
        self.sim = Simulation(scene_path(config))
        self.model, self.data = self.sim.model, self.sim.data
        self.locomotion = config.task == "go2_velocity"
        self.frame_skip = round(config.control_dt / self.model.opt.timestep)
        if not np.isclose(self.frame_skip * self.model.opt.timestep, config.control_dt):
            raise ValueError("control_dt must be a multiple of the physics timestep")
        self.actuated = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[self.actuated]
        self.dofs = self.model.jnt_dofadr[self.actuated]
        self.home = self.data.ctrl.copy()
        if self.locomotion:
            configure_servos(self.sim)
        else:
            from ..scenes.aloha_task import configure_grasp_contacts

            configure_grasp_contacts(self.sim)
            self.arm_actuators = np.array(
                [
                    i
                    for i in range(self.model.nu)
                    if "gripper" not in self.model.actuator(i).name
                ]
            )
            self.gripper_actuators = np.array(
                [
                    i
                    for i in range(self.model.nu)
                    if "gripper" in self.model.actuator(i).name
                ]
            )
            self.arm_dofs = self.dofs[self.arm_actuators]
            self.sim.before_step = self._gravity_compensation
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.targets = np.zeros((2, 3))
        self.command = 0.3
        self.elapsed_steps = 0
        self.previous_action = np.zeros(self.model.nu)
        obs = self._observation()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=obs.shape, dtype=np.float32
        )
        self._needs_reset = True

    def _gravity_compensation(self):
        self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]

    def _tools(self):
        return np.array(
            [self.data.site(side + "/gripper").xpos for side in ["left", "right"]]
        )

    def _objects(self):
        return np.array([self.data.body(f"object_{i}").xpos for i in range(2)])

    def _observation(self):
        state = [
            self.data.qpos[self.qadr],
            0.1 * self.data.qvel[self.dofs],
            self.data.ctrl,
            self.previous_action,
        ]
        if self.locomotion:
            body = self.data.body("base")
            rotation = body.xmat.reshape(3, 3)
            phase = 2 * np.pi * 2 * self.elapsed_steps * self.config.control_dt
            state.extend(
                [
                    rotation.T @ np.array([0.0, 0.0, -1.0]),
                    rotation.T @ self.data.qvel[:3],
                    self.data.qvel[3:6],
                    [body.xpos[2], self.command, np.sin(phase), np.cos(phase)],
                ]
            )
        else:
            state.extend(
                [
                    self._tools().ravel(),
                    self._objects().ravel(),
                    self.targets.ravel(),
                    (self.targets - self._tools()).ravel(),
                ]
            )
        return np.concatenate(state).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.elapsed_steps = 0
        self.previous_action[:] = 0
        self._needs_reset = False
        self.command = float(self.np_random.uniform(0.2, 0.45))
        if self.locomotion:
            self.data.qpos[self.qadr] += self.np_random.uniform(
                -0.015, 0.015, self.model.nu
            )
            mujoco.mj_forward(self.model, self.data)
        elif self.config.task == "aloha_reach":
            scratch = mujoco.MjData(self.model)
            for _ in range(32):
                scratch.qpos[:] = self.data.qpos
                scratch.qpos[self.qadr[self.arm_actuators]] += self.np_random.uniform(
                    -0.25, 0.25, len(self.arm_actuators)
                )
                mujoco.mj_forward(self.model, scratch)
                self.targets = np.array(
                    [
                        scratch.site(side + "/gripper").xpos.copy()
                        for side in ["left", "right"]
                    ]
                )
                if np.min(np.linalg.norm(self.targets - self._tools(), axis=1)) > 0.045:
                    break
        else:
            self.targets = np.array([[-0.14, 0.23, 0.03], [0.14, 0.23, 0.03]])
            for i in range(2):
                joint = self.model.body(f"object_{i}").jntadr[0]
                address = self.model.jnt_qposadr[joint]
                self.data.qpos[address : address + 2] += self.np_random.uniform(
                    -0.015, 0.015, 2
                )
            mujoco.mj_forward(self.model, self.data)
        self.start_x = float(self.data.body("base").xpos[0]) if self.locomotion else 0.0
        self.previous_distance = self._distance()
        return self._observation(), self._info(False, {})

    def _distance(self):
        if self.locomotion:
            return 0.0
        positions = (
            self._tools() if self.config.task == "aloha_reach" else self._objects()
        )
        return float(np.mean(np.linalg.norm(positions - self.targets, axis=1)))

    def _info(self, success, terms):
        return {
            "is_success": bool(success),
            "reward_terms": terms,
            "distance": self._distance(),
            "sim_time": float(self.data.time),
            "control_steps": self.elapsed_steps,
            "max_penetration": max(0.0, -float(min(self.data.contact.dist, default=0))),
        }

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset() is required before stepping a new episode")
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Action must contain one finite value per actuator")
        action = np.clip(action, -1, 1)
        if self.locomotion:
            elapsed = self.elapsed_steps * self.config.control_dt
            ramp = min(elapsed, 1.0)
            nominal = (
                self.home * (1 - ramp)
                + gait_targets(elapsed, stride=0.2 * self.command / 0.4) * ramp
            )
            targets = nominal + 0.15 * action
        else:
            targets = self.data.ctrl.copy()
            targets[self.arm_actuators] += 0.04 * action[self.arm_actuators]
            limits = self.model.actuator_ctrlrange[self.gripper_actuators]
            targets[self.gripper_actuators] = limits[:, 0] + (
                action[self.gripper_actuators] + 1
            ) * 0.5 * (limits[:, 1] - limits[:, 0])
        self.data.ctrl[:] = np.clip(
            targets,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self.sim.step(self.frame_skip)
        self.elapsed_steps += 1
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        terms = {
            "action_cost": -0.005 * float(np.mean(action**2)),
            "smoothness": -0.005 * float(np.mean((action - self.previous_action) ** 2)),
        }
        if self.locomotion:
            body = self.data.body("base")
            rotation = body.xmat.reshape(3, 3)
            velocity = rotation.T @ self.data.qvel[:3]
            upright = float(rotation[2, 2])
            failure = body.xpos[2] < 0.16 or upright < 0.6 or abs(body.xpos[1]) > 0.65
            # The first corridor is 8 m long. End at 3 m before reaching its wall.
            success = body.xpos[0] - self.start_x > 3.0
            terms.update(
                velocity=float(np.exp(-(((velocity[0] - self.command) / 0.25) ** 2))),
                upright=0.2 * upright,
                lateral=-0.2 * float(velocity[1] ** 2),
                angular=-0.02 * float(np.sum(self.data.qvel[3:6] ** 2)),
                fall=-3.0 * float(failure),
            )
        else:
            distance = self._distance()
            positions = (
                self._tools() if self.config.task == "aloha_reach" else self._objects()
            )
            errors = np.linalg.norm(positions - self.targets, axis=1)
            success = bool(np.all(errors < 0.025))
            failure = bool(np.any(self._objects()[:, 2] < -0.1))
            terms.update(
                progress=5.0 * (self.previous_distance - distance),
                proximity=0.2 * float(np.exp(-8 * distance)),
                success=5.0 * float(success),
            )
            if self.config.task == "aloha_pick_place":
                terms.update(
                    approach=0.1
                    * float(
                        np.exp(
                            -10
                            * np.mean(
                                np.linalg.norm(self._tools() - self._objects(), axis=1)
                            )
                        )
                    ),
                    lift=float(np.mean(np.clip(self._objects()[:, 2] - 0.03, 0, 0.15))),
                )
            self.previous_distance = distance
        self.previous_action = action.copy()
        terminated = bool(success or failure or not finite)
        truncated = bool(
            not terminated and self.elapsed_steps >= self.config.max_episode_steps
        )
        self._needs_reset = terminated or truncated
        observation = np.nan_to_num(
            self._observation(), nan=0.0, posinf=100.0, neginf=-100.0
        )
        reward = float(sum(terms.values())) if finite else -10.0
        return observation, reward, terminated, truncated, self._info(success, terms)
