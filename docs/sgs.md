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

Two pilots on `rod_16`, sparse reward, N = 512, 8 workers, one seed each. Held-out: 60 fresh
configurations, 20 per strategy, Wilson 95% intervals.

| Steps | Sampler | Held-out all | Near goal | Hovering | Reaching |
| --- | --- | --- | --- | --- | --- |
| 0.6M | Uniform | 4/60 (0.03–0.16) | 4/20 | 0/20 | 0/20 |
| 0.6M | SGS, κ = 1 | 4/60 (0.03–0.16) | 4/20 | 0/20 | 0/20 |
| 1.5M | Uniform | 15/60 (0.16–0.37) | 8/20 | 6/20 | 1/20 |
| 1.5M | SGS, κ = 5 | 14/60 (0.14–0.35) | 7/20 | 7/20 | 0/20 |

Where the data went at the end of the 1.5M-step runs (share of resets, training success
on the buffer):

| Configurations | Uniform share | Uniform success | SGS κ = 5 share | SGS success |
| --- | --- | --- | --- | --- |
| Near goal, fraction 0–0.3 (part already in the hole) | 0.11 | 0.92 | 0.03 | 0.96 |
| Near goal, 0.3–0.6 (chamfer zone) | 0.09 | 0.44 | 0.13 | 0.53 |
| Near goal, 0.6–1.1 | 0.13 | 0.35 | 0.23 | 0.39 |
| Hovering | 0.33 | 0.33 | 0.55 | 0.40 |
| Reaching | 0.33 | 0.00 | 0.06 | 0.00 |

Training success over the run, by start strategy (rolling sixths of the episodes):

| | Near goal | Hovering | Reaching |
| --- | --- | --- | --- |
| Uniform | 0.35 → 0.75 | 0.14 → 0.51 | 0.00 → 0.00 |
| SGS κ = 5 | 0.40 → 0.74 | 0.18 → 0.71 | 0.00 → 0.01 |

Reading:

- The sampler does what the formula says. With κ = 5 it takes resets away from the two
  ends, seated starts (solved) and reaching (never solved), and hands them to hovering and
  the upper half of the insertion path. Reaching falls from a third of all resets to 6%.
- At 0.6M steps nothing is learned from a hover by either policy; at 1.5M both insert
  from a hover about a third of the time, and the held-out intervals overlap completely.
  SGS's higher training success on hovering starts (0.71 vs 0.51) does not carry to fresh
  hovering configurations (7/20 vs 6/20): the buffer configurations it keeps revisiting are
  the ones it has learned, which is the mastered end of the kernel showing up at N = 512.
- Reaching stays at zero for both. Nothing in 1.5M sparse-reward steps discovers
  grasp, lift, carry, and insert in sequence; this is the part of the paper that lives at
  the 2^20-environment scale and is not reproduced here.

So, on this budget: no measurable difference between SGS and uniform on held-out
configurations, with one seed, which is consistent with the paper's own small-scale rows.
What would move it: three seeds per sampler, the linear curriculum as the third arm, and a
budget where reaching starts to succeed at all (an order of magnitude more steps, or the
mjlab backend on a GPU). The code is ready for that; the compute is not spent yet.
