# Success-Guided Sampling at small scale

A reproduction of the sampler from *A Balanced Data Diet: Addressing the Exploration
Bottleneck in Mega-Scale RL for Robot Control* (Zhang et al., CoRL 2026,
[arXiv 2610.12465](https://arxiv.org/abs/2610.12465), [site](https://sgs-rl.github.io/)),
run on the NIST board in this repository with CPU MuJoCo instead of a million GPU
environments. The paper's claim is about scale; the question here is narrower:

> With 8 environments, sparse reward, and the same NIST insertion tasks, does
> choosing *which configuration to reset to* already change what PPO learns?

## What SGS is

Standard PPO plus an outer loop over a fixed buffer of task configurations
(initial state, goal, scene). Each configuration keeps its last `H` outcomes; its
success rate `p_i` is mapped through a Beta-shaped kernel peaked at a target rate `t`,

```
w_i = (p_i + ε)^(κ t) · (1 − p_i + ε)^(κ (1 − t)),      P(i) ∝ exp(log(w_i + ε) / T)
```

so configurations the policy never solves and configurations it has mastered both
receive little data, and the frontier receives most of it. Manipulation values in the
paper: `N = 32768, t = 0.5, κ = 1, ε = 1e-4, T = 2, H = 100`.

Resets for assembly are split equally over three strategies: **Near-Goal** (part along
the assembly path, insertion fraction in `[0, 1.1]`), **Stable Grasp** (part held above
the target), and **Reaching** (part anywhere in the workspace, gripper aperture random).
The reward is sparse: `+100` on success, `−1` on failure, and `1e-5 … 1e-4` penalties on
action magnitude, action rate, and joint speed.

## What is implemented here

| Paper | Here |
| --- | --- |
| Isaac Lab, up to 2^20 environments | MuJoCo, 8 SubprocVecEnv workers |
| N = 32768 configurations, H = 100 | N = 512, H = 20 (25k episodes per run would leave 32768 configurations mostly unvisited) |
| Point-cloud teacher observation | Proprioception, part pose relative to the TCP, wrench on the part |
| Δpose 6 + gripper, OSC | Δposition 3 + Δyaw + gripper, damped least-squares IK onto position servos |
| Nut, rod, gear, BNC, waterproof, rectangular peg | Rods and rectangular pegs (`rod_4`–`rod_16`, `peg_4`–`peg_16`) |
| Baselines: uniform, linear curriculum, PLR | Uniform, linear curriculum |

The hole offset (±5 mm) is hidden from the policy in every strategy; the in-hand
offset and yaw are visible through the part pose.

Code: `rl/configs.py` (buffer), `rl/insert.py` (three start strategies, sparse reward),
`rl/sgs.py` (samplers and the `VecEnvWrapper` that assigns the next configuration to
each worker one episode ahead and feeds outcomes back), `rl/sgs_report.py`.

```bash
uv sync --extra rl
uv run --extra rl mujoco-viser-train --config assets/rl/sgs/rod16_sgs.json --runs runs/sgs
uv run --extra rl mujoco-viser-train --config assets/rl/sgs/rod16_uniform.json --runs runs/sgs
uv run --extra rl python -m mujoco_viser_usage.rl.evaluate --checkpoint <run>/checkpoints/final.zip --episodes 60 --run-dir <run>-eval
uv run --extra rl python -m mujoco_viser_usage.rl.sgs_report runs/sgs/*-eval
```

Evaluation draws 60 fresh configurations (20 per strategy) the sampler never saw.
The RL dashboard shows a **Sampler** panel for such runs: sampling share and success
rate per strategy over training, from `sampler.jsonl`.

## Results

Pilot, `rod_16`, sparse reward, 512 configurations, 8 workers, 600k steps, one seed.
Held-out: 60 fresh configurations, 20 per strategy.

| Sampler | Held-out all | Near goal | Hovering | Reaching | Training success, last 20% |
| --- | --- | --- | --- | --- | --- |
| Uniform | 4/60 (CI 0.03–0.16) | 4/20 | 0/20 | 0/20 | 0.20 |
| SGS (t 0.5, κ 1, T 2, H 20) | 4/60 (CI 0.03–0.16) | 4/20 | 0/20 | 0/20 | 0.34 |

Where the data went (sampler state at the end of the SGS run):

| Configurations | Share of resets | Success rate |
| --- | --- | --- |
| Near goal, fraction 0–0.3 (part already in the hole) | 0.11 | 0.94 |
| Near goal, 0.3–0.6 (chamfer zone) | 0.15 | 0.21 |
| Near goal, 0.6–1.1 | 0.15 | 0.09 |
| Hovering | 0.46 | 0.13 |
| Reaching | 0.13 | 0.00 |

Uniform keeps 0.33 per strategy by construction.

Reading: SGS did what the formula says. Reaching, never solved, dropped from a third of
the resets to an eighth; the chamfer zone and hovering starts got the data. The higher
training success is a distribution effect, not a better policy: on fresh configurations
both policies seat the part only when it starts inside the hole, and neither inserts from
a hover, which the dense-reward hovering-only policy in the README does 10/10.

So 600k sparse-reward steps on 8 CPU workers is below the budget where any sampler
matters, which is consistent with the paper's own tables: at 4K environments SGS and
uniform are both near zero on assembly. With `κ = 1, T = 2` the frontier is weighted only
about 7× over solved or unsolved configurations, so with `N = 512` the sampler stays close
to uniform. The next experiment is the one that can separate the two:

- `κ = 5` (the paper's locomotion value), which sharpens the kernel enough to matter at
  this `N`;
- 2M steps, three seeds per sampler, uniform / linear / SGS;
- evaluation on 60 held-out configurations per checkpoint, Wilson 95% intervals.

That is roughly six hours of CPU on this machine; it has not been run yet.
