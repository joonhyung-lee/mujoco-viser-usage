"""Task-configuration samplers: uniform, linear curriculum, Success-Guided Sampling.

SGS (Zhang et al., CoRL 2026): each configuration keeps its last H outcomes;
its weight is a Beta-shaped kernel peaked at the target success rate t,
    w_i = (p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)),
and resets draw from softmax(log(w + eps) / T). Unsolved and mastered
configurations both get little data; the frontier gets most of it.
"""

from collections import deque

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from .configs import STRATEGIES, InsertConfig
from .storage import append_json, write_npz


class Sampler:
    def __init__(self, configs: list[InsertConfig], seed=0):
        self.configs = configs
        self.rng = np.random.default_rng(seed)
        self.successes = np.zeros(len(configs))
        self.visits = np.zeros(len(configs))
        self.progress = 0.0  # fraction of the training budget consumed

    def probabilities(self):
        return np.full(len(self.configs), 1 / len(self.configs))

    def sample(self):
        return int(self.rng.choice(len(self.configs), p=self.probabilities()))

    def update(self, index, success):
        self.visits[index] += 1
        self.successes[index] += success

    def state(self):
        return {
            "probabilities": self.probabilities(),
            "success_rate": self.successes / np.maximum(self.visits, 1),
            "visits": self.visits,
            "configs": np.stack([c.as_array() for c in self.configs]),
        }


class UniformSampler(Sampler):
    pass


class LinearSampler(Sampler):
    """Hand-made curriculum: seated starts first, then hovering, then reaching."""

    def probabilities(self):
        eligible = np.array(
            [
                (c.strategy == "near_goal" and c.fraction <= 0.3 + 0.8 * self.progress)
                or (c.strategy == "stable" and self.progress > 0.5)
                or (c.strategy == "reaching" and self.progress > 0.75)
                for c in self.configs
            ]
        )
        if not eligible.any():
            eligible[:] = True
        return eligible / eligible.sum()


class SuccessGuidedSampler(Sampler):
    def __init__(
        self,
        configs,
        seed=0,
        *,
        target=0.5,
        kappa=1.0,
        eps=1e-4,
        temperature=2.0,
        history=20,
    ):
        super().__init__(configs, seed)
        self.target, self.kappa, self.eps, self.temperature = (
            target,
            kappa,
            eps,
            temperature,
        )
        self.history = [deque(maxlen=history) for _ in configs]

    def update(self, index, success):
        super().update(index, success)
        self.history[index].append(float(success))

    def probabilities(self):
        p = np.array([np.mean(h) if h else 0.0 for h in self.history])
        w = (p + self.eps) ** (self.kappa * self.target) * (1 - p + self.eps) ** (
            self.kappa * (1 - self.target)
        )
        logits = np.log(np.maximum(w, self.eps) + self.eps) / max(self.temperature, 1.0)
        logits -= logits.max()
        weights = np.exp(logits)
        return weights / weights.sum()


SAMPLERS = {
    "uniform": UniformSampler,
    "linear": LinearSampler,
    "sgs": SuccessGuidedSampler,
}


class ConfigSampling(VecEnvWrapper):
    """Assigns each worker's next configuration and feeds outcomes back.

    Workers auto-reset inside step, so an assignment is consumed one episode
    after it is made; the sampling distribution moves slowly enough for that.
    """

    def __init__(self, venv, sampler: Sampler, run, total_timesteps, every=50):
        super().__init__(venv)
        self.sampler = sampler
        self.run = run
        self.total = total_timesteps
        self.every = every
        self.steps = 0
        self.episodes = 0

    def _assign(self, indices):
        for i in indices:
            self.venv.env_method("assign", self.sampler.sample(), indices=[i])

    def reset(self):
        self._assign(range(self.num_envs))
        obs = self.venv.reset()
        self._assign(range(self.num_envs))
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.steps += self.num_envs
        self.sampler.progress = min(self.steps / self.total, 1.0)
        for i in np.flatnonzero(dones):
            if infos[i].get("config", -1) >= 0:
                self.sampler.update(infos[i]["config"], infos[i]["is_success"])
            self.episodes += 1
            self._assign([i])
            if self.episodes % self.every == 0:
                self.save()
        return obs, rewards, dones, infos

    def save(self):
        state = self.sampler.state()
        write_npz(self.run / "sampler.npz", steps=np.array(self.steps), **state)
        strategies = state["configs"][:, 0]
        record = {"steps": self.steps, "episodes": self.episodes}
        for k, name in enumerate(STRATEGIES):
            mask = strategies == k
            record[f"share_{name}"] = float(state["probabilities"][mask].sum())
            record[f"success_{name}"] = float(state["success_rate"][mask].mean())
        append_json(self.run / "sampler.jsonl", record)

    def close(self):
        self.save()
        self.venv.close()
