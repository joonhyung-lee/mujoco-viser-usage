# mjlab integration

`mujoco-viser-usage` keeps its own task assets and experiment interface. mjlab owns
parallel physics and RSL-RL training. The dashboard supports both native mjlab and the existing CPU/SB3 backend.
Their checkpoints are not interchangeable.

## Environment

The integration is isolated because mjlab 1.6.0 requires MuJoCo 3.11 while the
existing viewer environment uses a newer MuJoCo. Do not install mjlab into the
root virtual environment.

```sh
uv sync --project integrations/mjlab
```

The nested lockfile pins the complete environment. Use Linux/NVIDIA CUDA for
substantial training. This pinned version also executes CPU smoke tests on macOS;
that does not establish GPU throughput or policy quality.

## Dashboard

```sh
uv run mujoco-viser-rl --runs runs --port 8090
```

Open http://localhost:8090. Choose an ALOHA or Go2 scene, then **Training → Settings →
Backend → mjlab**. The launch button changes to **Train with mjlab**. Settings also
contains CPU/CUDA device, parallel worlds, and the training budget. Up to four worlds
are recorded for display, regardless of training batch size. The Go2 **Terrain**
choice applies to both backends; every Warp world shares the selected terrain.

Pause/resume acts at policy-step boundaries. Stop saves `stopped.pt`; Continue from
checkpoint starts a new run with the saved model/optimizer and the original task
configuration. Checkpoints are published atomically, so live evaluation cannot read
an unfinished write. Training budgets round up to complete PPO rollout batches;
the displayed step count reports actual transitions.

**Evaluate checkpoint · 3 episodes** runs scripted control and deterministic policy
inference on three held-out seeds. It leaves training selection intact and publishes
an **Open same-seed comparison** link after completion. Stop comparison cancels the
evaluation. The dashboard owns its evaluation/viewer processes and closes them when
it exits; existing unrelated servers are unaffected.

Training uses `integrations/mjlab/.venv/bin/python`. For an installed wheel or custom
environment, set `MUJOCO_VISER_MJLAB_PYTHON` to that environment's Python executable.
The browser viewer stays in the root environment and loads the run's exported model,
including recorded mocap poses and contact-force vectors. SB3 history stays readable.

## Native tasks

| Task | Control | Episode limit |
| --- | --- | --- |
| `Viser-Aloha-Reach-v0` | Both arms, absolute home offsets | 8 s |
| `Viser-Aloha-Lift-v0` | Left arm, incremental joint targets | 12 s |
| `Viser-Aloha-Place-v0` | Left arm, incremental joint targets | 22 s |
| `Viser-Aloha-Bimanual-v0` | Both arms, incremental joint targets | 42 s |
| `Viser-Go2-Velocity-v0` | 12 joint residuals over the scripted trot | 10 s |

All tasks use physical MuJoCo-Warp steps: 2 ms physics, 20 ms control. Objects are
free bodies, gripper force is limited to 4 N per actuator, and arm gravity/bias
compensation is applied each physics substep. Only reset writes initial states.
Scripted control uses scratch-data IK, then the same action interface as the policy.

Lift/place/bimanual success requires a sustained physical lift; place additionally
requires object proximity and open grippers. Reach goals come from a deterministic
bank of forward-kinematics samples. Object positions and goal choices are seeded.

The Go2 task mirrors the SB3 `go2_velocity` environment: the XML motors are replaced
by mjlab position servos (kp 150, kd 4, XML torque limits), the action is a ±0.15 rad
residual on the analytic trot, and the forward command is seeded in 0.2–0.45 m/s.
Success is 3 m of forward travel; falling, tipping, or leaving the first corridor
terminates the episode. The bundled maze or terrain XML (`ramps`, `steps`, `rough`,
`mixed`) is copied into every world as static geometry, including the ramp mesh.
Foot contacts are recorded per foot in world coordinates. The scripted baseline
is the trot with zero residual.

## Train

Small CPU integration smoke test (not a trained manipulation policy):

```sh
uv run --project integrations/mjlab python -m mujoco_viser_usage.mjlab train \
  Viser-Aloha-Reach-v0 --gpu-ids '[]' --env.scene.num-envs 2 \
  --agent.num-steps-per-env 8 --agent.max-iterations 2 \
  --agent.algorithm.num-mini-batches 2 --agent.algorithm.num-learning-epochs 1 \
  --log-root runs/mjlab-smoke
```

Starting configuration for a GPU run; tune environment count to available memory:

```sh
uv run --project integrations/mjlab python -m mujoco_viser_usage.mjlab train \
  Viser-Aloha-Reach-v0 --gpu-ids '[0]' --env.scene.num-envs 512 \
  --agent.max-iterations 1000 --log-root runs/mjlab
```

`Viser-Go2-Velocity-v0` accepts the same arguments; the CLI registration uses the
flat maze, while the dashboard passes the selected terrain.

Native mjlab arguments, including resume, remain available through `--help`.
TensorBoard is the default logger. Runs contain native `.pt` checkpoints,
`agent.json`, `environment.pkl`, `run.json`, a self-contained `scene/`, and traces
for at most four worlds. `environment.pkl` and checkpoints are local Python
experiment artifacts: load artifacts from trusted runs.

## Compare a checkpoint with scripted control

Use the timestamped run directory printed by training:

```sh
uv run --project integrations/mjlab python -m mujoco_viser_usage.mjlab.evaluate \
  runs/mjlab-smoke/viser_aloha/<timestamp> --seeds 1000003 1000004
```

Evaluation uses the saved environment configuration, one world, deterministic
policy inference, and explicit resets. It verifies exact initial qpos and goals
match between scripted and policy runs. The newest numbered checkpoint is chosen
unless `--checkpoint PATH` is supplied. It writes a new `runs/*mjlab-compare*/`
directory and `comparison.json` with checkpoint SHA-256, outcomes, force/height
measurements, and event timestamps. Run with the same code revision as training.

```sh
uv run python -m mujoco_viser_usage.rl.comparison \
  runs/<comparison-directory> --port 8091
```

Open http://localhost:8091. Both views share the same simulation-time cursor.
Pause, scrub, restart, select a seed, or jump to a recorded event. A shorter episode
holds its terminal pose while the other finishes. Details holds summary metrics,
playback speed, and contact-force visibility. EEF trails and force arrows use
separate namespaces for each view.

The upper-left **Replay RGB** panel shows the same camera for scripted control (left)
and policy (right). Choose left wrist, right wrist, or overview. Both images are
rendered from recorded poses using the exported MJCF; these are reconstructed replay
images, not sensor recordings captured during training. The captions show each
image's actual simulation time, including when a shorter episode holds its terminal
pose. Seeking or changing camera invalidates pending older images. Disable the panel
under Details when not needed.

Rendering uses one dedicated thread and updates at most 10 Hz. A missing OpenGL
backend displays an RGB error while the 3D replay remains usable.

New reach runs use nonpositive distance shaping plus a fixed +2 success bonus,
so delaying completion cannot accumulate extra positive reward. Previously saved
environment configurations retain their original positive shaping. Compare return
only within the same reward definition; inspect success, duration, and goal error.
Lowering after lift is recorded as an observation, not automatically classified
as a failed grasp: intentional placement also lowers the object.

## Recording semantics

Terminal states are captured **before** mjlab's automatic reset. Initial and
terminal frames are always retained; ordinary training frames are sampled every
five control steps. Evaluation records every control step. Interrupted training
episodes remain explicitly marked `interrupted`.

Traces include qpos/qvel/ctrl, applied forces, mocap pose, actions, targets, EEF and
object positions, and weighted reward terms. Contact sensors retain the strongest
contact per object in world coordinates; these are not all contacts or net grasp
forces. As in mjlab termination/reward evaluation, terminal derived quantities
can lag qpos by one 2 ms physics substep. Force/penetration summaries report sampled
values; they do not claim substep maxima.

The exported scene includes mesh bytes and actual runtime solver options. Playback
uses MuJoCo forward kinematics solely to display recorded state; it does not advance
or alter a training environment. Warp float32 qpos is converted to float64 for
native MuJoCo quaternion interpolation.

## Validation

```sh
uv run --project integrations/mjlab --with pytest python -m pytest tests/test_mjlab.py -q
uv run pytest tests/test_mjlab_dashboard.py tests/test_playback.py tests/test_dashboard_scenes.py tests/test_overlays.py -q
```

Physical baseline verification (longer than the smoke tests):

```sh
uv run --project integrations/mjlab python -m mujoco_viser_usage.tools.check_mjlab_baselines
```

On the local CPU, seed 1000003 completed reach, lift, place, and bimanual through
physical stepping. This validates task solvability, not learned-policy performance.
The two-update PPO smoke checkpoint did not solve reach. For Go2, the open-loop trot
reached 3 m after 9.5 s on the same CPU seed; the script only asserts that it advances.

Native tests cover all four task resets/steps, deterministic initial states, correct
home controls, independent arm targets, incremental action range, terminal trace
retention, and exported scene loading with matching solver options.

Remaining integration work: broader failure classification, GPU learning validation,
and Husky task migration. The comparison
viewer opens in a separate tab; it is launched and owned by the dashboard.
