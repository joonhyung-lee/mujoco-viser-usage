"""Damped least-squares site IK on scratch data; never teleports live physics."""

import mujoco
import numpy as np


class SiteIK:
    def __init__(self, model: mujoco.MjModel, site: str, joints: list[str]) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        self.joints = np.array([model.joint(name).id for name in joints])
        self.qadr = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]

    def solve(
        self,
        qpos: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        iterations: int = 300,
    ) -> np.ndarray:
        self.data.qpos[:] = qpos
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            current = self.data.site_xmat[self.site].reshape(3, 3)
            error = np.concatenate(
                (
                    position - self.data.site_xpos[self.site],
                    0.5
                    * sum(np.cross(current[:, i], rotation[:, i]) for i in range(3)),
                )
            )
            if np.linalg.norm(error[:3]) < 2e-5 and np.linalg.norm(error[3:]) < 1e-4:
                return self.data.qpos[self.qadr].copy()
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack((jp[:, self.dofs], jr[:, self.dofs]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            self.data.qpos[self.qadr] += np.clip(delta, -0.15, 0.15)
            for jid, qadr in zip(self.joints, self.qadr):
                if self.model.jnt_limited[jid]:
                    self.data.qpos[qadr] = np.clip(
                        self.data.qpos[qadr], *self.model.jnt_range[jid]
                    )
        raise ValueError(f"IK did not converge for {position.tolist()}")
