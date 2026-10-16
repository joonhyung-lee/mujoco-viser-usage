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

`rod_16`, sparse reward, N = 512 configurations, 8 workers, 1.5M steps. Three seeds per
sampler (the buffer is regenerated from the seed). Held-out: 60 fresh configurations,
20 per strategy, the same set for every run. Pooled Wilson 95% intervals.

| Sampler | Per seed, /60 | Pooled | Near goal | Hovering | Reaching | Episodes per run |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 15, 6, 10 | 0.17 (0.12–0.23) | 0.38 | 0.12 | 0.02 | ~9.7k |
| Linear curriculum | 27, 22, 30 | 0.44 (0.37–0.51) | 0.67 | 0.65 | 0.00 | ~20.7k |
| SGS (t 0.5, κ 5, T 2, H 20) | 14, 26, 31 | 0.39 (0.33–0.47) | 0.57 | 0.62 | 0.00 | ~12k |

An earlier pilot at 0.6M steps with κ = 1 gave 4/60 for both uniform and SGS: below the
budget where any sampler matters.

Where the data went at the end of a 1.5M-step SGS run (seed 12; share of resets, training
success on the buffer):

| Configurations | Uniform share | SGS share | SGS success |
| --- | --- | --- | --- |
| Near goal, fraction 0–0.3 (part already in the hole) | 0.11 | 0.03 | 0.96 |
| Near goal, 0.3–0.6 (chamfer zone) | 0.09 | 0.13 | 0.53 |
| Near goal, 0.6–1.1 | 0.13 | 0.23 | 0.39 |
| Hovering | 0.33 | 0.55 | 0.40 |
| Reaching | 0.33 | 0.06 | 0.00 |

Reading:

- **Where episodes reset decides what gets learned.** Same PPO, reward, and step budget;
  the sampler alone moves held-out success from 0.17 to 0.39–0.44. The uniform interval
  does not overlap either of the others.
- **SGS and the hand-made curriculum tie.** Their intervals overlap (0.33–0.47 vs
  0.37–0.51). The linear schedule encodes knowledge of this task: start seated, widen the
  insertion fraction, then add hovering and reaching. SGS has none of that and gets to the
  same place by reading success rates, which is the paper's argument for it. It pays for
  the discovery: with N = 512 and H = 20 its estimates are noisy, and its kernel also
  discounts the seated starts that are the only early positive signal, so it reaches its
  final distribution later than the schedule does.
- **Episode count is the mechanism.** Near-goal starts end quickly, so the linear run saw
  about twice as many episodes as the uniform run in the same number of steps, and
  therefore about twice as many terminal rewards. SGS lands in between.
- **Reaching stays at zero for all three.** Grasp, lift, carry, and insert under a sparse
  reward is not discovered in 1.5M steps by any sampler. That is the regime the paper
  reaches with 2^20 environments and is not reproduced here.
- **The paper's "linear stalls at scale" is outside this range.** It appears from tens of
  thousands of environments; these runs use 8.

Single-run variance is large (uniform 6–15, SGS 14–31 of 60), so three seeds support the
ordering, not the exact ratios.
