"""Deterministic scripted controllers, using the same actions as the RL policy."""

import mujoco
import numpy as np
import torch

from ..kinematics import SiteIK
from .mdp import state


def make_controller(env):
    if env.cfg.events["reset"].params["stage"] == "go2":
        from .go2 import ScriptedTrot

        return ScriptedTrot(env)
    return ScriptedController(env)


class ScriptedController:
    """One evaluation world; solve on scratch data, move only through env.step."""

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("Scripted evaluation requires one world")
        self.env = env
        self.term = env.action_manager.get_term("servo")
        self.stage = state(env).stage
        model = env.sim.mj_model
        qpos = env.sim.data.qpos[0].cpu().numpy().copy()
        controls = env.sim.data.ctrl[0].cpu().numpy().copy()
        self.phases = []
        self.elapsed = 0.0
        self.index = 0
        self.initial = controls.copy()
        for arm, side in enumerate(("left", "right")):
            if arm and self.stage in {"lift", "place"}:
                break
            actuator_ids = np.arange(arm * 7, arm * 7 + 6)
            joints = [
                model.joint(int(j)).name for j in model.actuator_trnid[actuator_ids, 0]
            ]
            solver = SiteIK(model, f"robot/{side}/gripper", joints)
            if self.stage == "reach":
                target = state(env).goals[0, arm].cpu().numpy()
                solver.data.qpos[:] = qpos
                jac = np.zeros((3, model.nv))
                for _ in range(100):
                    mujoco.mj_forward(model, solver.data)
                    error = target - solver.data.site_xpos[solver.site]
                    if np.linalg.norm(error) < 1e-5:
                        break
                    mujoco.mj_jacSite(model, solver.data, jac, None, solver.site)
                    j = jac[:, solver.dofs]
                    delta = j.T @ np.linalg.solve(j @ j.T + 1e-5 * np.eye(3), error)
                    solver.data.qpos[solver.qadr] += np.clip(delta, -0.05, 0.05)
                controls[actuator_ids] = solver.data.qpos[solver.qadr]
                continue
            obj = env.sim.data.xpos[0, state(env).body_ids[arm]].cpu().numpy().copy()
            goal = state(env).goals[0, arm].cpu().numpy().copy()
            sign = 1 if arm == 0 else -1
            rotation = np.array([[0, 0, sign], [0, sign, 0], [-1, 0, 0]])
            phases = [
                ("Approach", obj + [0, 0, 0.16], 0.037, 3.0),
                ("Lower", obj + [0, 0, 0.006], 0.037, 2.5),
                ("Grasp", obj + [0, 0, 0.006], 0.002, 1.5),
                ("Lift", obj + [0, 0, 0.16], 0.002, 2.5),
            ]
            if self.stage != "lift":
                phases += [
                    ("Transfer", goal + [0, 0, 0.16], 0.002, 3.0),
                    ("Place", goal + [0, 0, 0.006], 0.002, 2.5),
                    ("Release", goal + [0, 0, 0.006], 0.037, 1.5),
                    ("Retract", goal + [0, 0, 0.16], 0.037, 2.0),
                ]
            for name, position, opening, duration in phases:
                q = solver.solve(qpos, np.asarray(position), rotation)
                qpos[solver.qadr] = q
                controls[actuator_ids] = q
                controls[arm * 7 + 6] = opening
                self.phases.append((f"{side}: {name}", controls.copy(), duration))
        if self.stage == "reach":
            self.phases = [("Reach", controls, 2.0)]

    @property
    def phase(self):
        return self.phases[self.index][0]

    def __call__(self):
        _, target, duration = self.phases[self.index]
        self.elapsed += self.env.step_dt
        fraction = np.clip(self.elapsed / duration, 0.0, 1.0)
        blend = fraction**3 * (10 - 15 * fraction + 6 * fraction * fraction)
        desired = self.initial + blend * (target - self.initial)
        dimension = self.term.action_dim
        if self.term.cfg.incremental:
            action = (desired - self.term.targets[0].cpu().numpy()) / 0.04
        else:
            action = (desired - self.term.home.cpu().numpy()) / 0.6
        for i in (6, 13):
            low, high = self.term.limits[i].cpu().numpy()
            action[i] = 2 * (desired[i] - low) / (high - low) - 1
        if fraction >= 1 and self.index + 1 < len(self.phases):
            self.initial = target.copy()
            self.index += 1
            self.elapsed = 0.0
        return torch.tensor(
            action[:dimension].clip(-1, 1), dtype=torch.float32, device=self.env.device
        )[None]
