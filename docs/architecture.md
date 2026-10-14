# Architecture

The main interface is `ViserParser`: load a model, set controls, step physics,
render, or run the interactive workspace. It owns server lifetime and supports a
context manager. Callers need not assemble a GUI, command queue, and render loop.

| Module | Responsibility | Does not own |
| --- | --- | --- |
| `viser_parser.py` | Public robotics interface, composition, lifetime | Mesh conversion, GUI widgets |
| `simulation.py` | MuJoCo model/data, reset, control validation, physics | Viser, network, wall clock |
| `runtime.py` | Serialized commands, paused/running state, fixed-step budget | GUI, rendering |
| `viewer.py` | Geometry conversion, scene handles, world poses, camera | Physics stepping |
| `controls.py` | User intent, display units, panel presentation | Direct physics mutation |
| `scene_style.py` | Colors, lighting, layout defaults | Model state |
| `app.py` | CLI arguments | Application logic |

## Patterns used for concrete problems

Following [Refactoring Guru's explanation](https://refactoring.guru/design-patterns/what-is-pattern),
patterns are adapted to the problem rather than copied as mandatory class hierarchies.

- **Facade:** `ViserParser` hides setup, ownership, and the interactive loop.
- **Command:** callbacks enqueue bound functions with captured arguments; the
  runtime applies them on the owner thread. Python callables avoid a class per button.
- **Observer:** Viser events report user intent; the panel reads a runtime snapshot.
  Programmatic control synchronization ignores events without a client, preventing
  reset feedback from overwriting fresh state.

No singleton or abstract factory is needed: there is one physics implementation
and one renderer. Internal modules compose directly at the facade.

## Ownership and timing

Model/data mutations happen on one owner thread. The interactive `run()` loop owns
physics progression; `step()` and `render()` are for an externally owned loop.
Do not run both loops concurrently. Direct model/data access is an explicit escape
hatch for controller experiments and has the same owner-thread requirement.
GUI callbacks never mutate MuJoCo directly. Camera reset is per client.

`render()` drains pending GUI commands but does not automatically advance the
runtime clock. In a manual loop, the caller controls stepping; the Start/Pause
button only affects the built-in `run()` loop. Prefer `run()` for full panel control.

Physics uses the model's fixed timestep and a fractional accumulator. Long wall-clock
stalls are capped at 100 ms. Resume/reset discard paused time. Slider commands do
not discard the integration budget. While paused with no changes, the loop sends
no repeated geometry updates. The scene's meshes are uploaded once.

## What was learned from mjviser

Reference checkout: `ae0b219a2049a95e1321214e4efd1d395ee4bf39` at
https://github.com/mujocolab/mjviser .

`Viewer` and `ViserMujocoScene` separate the active loop from scene rendering.
Its tabbed controls, fixed-step budget, and dirty refresh informed this refactor.
Our implementation stays independent: it keeps the Isaac-inspired scene and the
small parser interface without importing mjviser's larger GUI and renderer.
No mjviser source code is vendored or copied.

The current renderer is a deliberately smaller subset. mjviser's batched body
meshes, textures, and multi-environment rendering are
future adoption candidates, not features of this project today.

## Pick/place extension

`SiteIK` uses scratch `MjData` to plan poses; it never edits live physics.
`PickPlaceTask` owns named phases and smooth joint-space interpolation, with
per-step arm bias compensation and a separate RG2 opening target. The simulation
accepts optional `before_step` and `on_reset` callbacks so task behavior stays out
of its generic physics implementation. Reset cancels the active task.
The facade enables this task explicitly with `pick_place=True`; the CLI does so
for its default bundled pick/place scene. Other catalog controllers are composed by the task factory.

The example assumes the bundled UR5e actuator ordering and RG2 linkage. Its
success check requires an actual lift and placement within 4 cm of the target,
with the object resting at table height. This is a scripted manipulation example,
not a general collision-avoiding motion planner or a calibrated hardware model.

## Penetration regression (2026-10-14)

The first pick/place check only asserted lift and final placement; it missed deep
soft-contact penetration. The deterministic full trajectory measured a 22.580 mm
maximum at finger/object contact and 7.225 mm at object/table contact.

Isolated probes: reducing the RG2 joint actuator limit from ±15 to ±1 N·m reduced
maximum depth to 2.952 mm. Changing contact solver parameters alone to
`solref="0.004 1"`, `solimp="0.99 0.99 0.001"` reduced it to 0.320 mm. Together they
reduced maximum depth to 0.074 mm while preserving grasp, lift, and placement.
These are demo tuning values, not a calibrated RG2 hardware specification.

The regression now checks penetration below 1 mm at every physics step through
the complete task. `uv run python -m mujoco_viser_usage.tools.check_contacts` reports the worst
contact pairs, times, and phases and fails above that limit. The State folder shows
current maximum penetration; View can display contact points.

Physics remains `mj_step`, at 2 ms using `implicitfast`. `mj_forward` refreshes
transforms and derived values after integration; IK solves on separate data and
only supplies motor targets. Small residual penetration is permitted by soft
contact. This bound applies to the tested automatic trajectory, not arbitrary
manual commands. The RG2 linkage's internal self-collision is filtered; contacts
with the object and tables are enabled. Arm and gripper geometry are simplified
simulation models and do not constitute a full hardware collision model.

### Gentler closing torque

A follow-up sweep at 0.2, 0.35, 0.5, 0.75 and 1.0 N·m completed the default
trajectory at every value. The 0.35 N·m setting also passed with motion phase
durations reduced by 25%. Closing is now capped at 0.35 N·m, retaining 1 N·m
for opening (`forcerange="-0.35 1"`). This is actuator joint torque, not
a calibrated fingertip force. The final asymmetric-limit trajectory is checked
by the existing contact/placement regression and diagnostic script.

## Read-only overlays

`overlays.py` owns force arrows, measured trajectories, and the ego camera panel.
`contact_vectors` transforms the contact-frame wrench from `mj_contactForce` by
`contact.frame.T` into a world vector on geom2. A supported 1 kg sphere test checks
that the summed displayed support force is 9.81 N upward. Force arrows have an
explicit display scale and cap; they never apply forces to the simulation.

`TrajectoryOverlay` records the actual site position, bounds memory to 1,500
samples, updates its transparent tube at a lower rate, and distributes exactly
ten markers by cumulative path length. Markers use an opaque inner sphere and a
larger black back-face shell for a silhouette outline. Reset clears the history.

`EgoCamera` owns a MuJoCo Renderer and a Viser floating panel at viewport (16,16).
Rendering and resource teardown stay on the simulation owner thread. The facade
closes the renderer before the Viser server. Models without the ego camera do not
allocate a renderer. The RGB feed has a separate 10 Hz limit from scene updates.

The wrist camera uses a 180-degree optical-axis roll correction so its screen-up
axis projects positively onto world Z at the default top-down grasp pose. This
corrects inversion without changing the optical target or flipping RGB pixels.
Its upper-left panel is 420×315 CSS pixels with a 640×400 RGB render.

## Three scene domains

`scenes` is a small immutable catalog mapping a public scene ID to a packaged
MJCF, a description, and an optional concrete task. Manipulation, locomotion,
and navigation keep their model assets in separate subtrees. Spot and Spot+Arm
share assets. Simulation and rendering have no robot-name dispatch.

`scenes/maze.py` is the single occupancy-map definition. The development asset
builder emits collision walls and markers from it; the navigation task derives
a four-connected shortest route using BFS. The viewer draws the plan separately
from the measured site trajectory. No benchmark dependency or dataset download
is introduced. Rebuild bundled maze XMLs after editing the map.

`MazeNavigationTask` shares the existing start/reset/update/status lifecycle
with `PickPlaceTask`. It computes body heading from simulated rotation and writes
bounded wheel velocity targets. Wheel radius and track width come from the
model. All translation/rotation results from MuJoCo contacts; it never writes
live qpos or adds root forces. Goal arrival stops wheel targets. Timeout/tip-over
stops the task, and reset restores the initial state and cancels the task.

Locomotion scenes provide floating-base dynamics with position-servo controls.
Go2 adds the trot sample described below; Go1/Spot currently expose manual control.
Go2 has a position-servo variant because upstream torque motors cannot interpret
home joint angles as position targets. UR5e and ALOHA have automatic manipulation
samples; Franka exposes manual control. Husky uses the bundled source MJCF/STL
with URDF chassis inertia and cylinder wheel contacts. TurtleBot3 remains a primitive approximation.

## Scripted samples and automatic stepping

`ViserParser.from_scene()` composes the catalog scene and its concrete task through
`tasks.create_task()`. `Task` is a structural lifecycle protocol, so GUI controls
need no concrete controller imports. The existing `pick_place` and `navigation`
arguments remain compatible. Physics now starts by default at the facade/CLI;
`start_running=False` / `--paused` opts out. Reset still deliberately pauses.
`demo=True` / `--demo` also starts the scripted task. Headless `step()` remains explicit.

ALOHA uses scratch-data site IK for each arm, smooth joint targets, bias compensation
and bounded gripper forces. Its two objects move only through contacts. A pair of
measured EEF trajectories uses separate scene namespaces. Object/table contacts
use firmer solver settings; gripper force is limited to ±4 N in the task model.

Go2 uses a diagonal trot phase and two-link planar leg IK, with position servo
kp=150, kv=4 while retaining the original torque limits. It advances along the
first corridor, brakes after 2 m and checks for falls/timeouts. It is a scripted
walking sample, not a learned general maze-navigation policy.

## RL process boundaries

See `docs/rl.md` for commands, task definitions and artifact layouts. The Viser
control room owns a `RunSupervisor`, which launches a trainer in a new process
session. The trainer owns a shared SB3 PPO policy and `SubprocVecEnv` children
using the spawn start method. Each child owns its own `RobotEnv` model/data and
an `EpisodeRecorder`. UI callbacks enqueue control-room commands; process control
is a small atomic command file read at rollout boundaries/steps.

Training and rendering share no mutable MuJoCo objects. Workers publish atomic
state snapshots and append-only episode summaries. The control room reconstructs
read-only display state using `mj_forward` on separate data. A four-agent overview
translates display roots into tiles; worker physics is never translated. Replay
uses recorded states and resets trajectory history when switching workers/modes.

Checkpoints contain policy and optimizer state. Resume creates a new run and fresh
environment episodes; it does not claim to restore a partially collected rollout.
The evaluator uses deterministic policy inference on different seeds. Initial and
final policies, PPO metrics, explicit worker PIDs and replay trajectories make the
training implementation observable and testable without claiming convergence.


## Terrain and paced rollout display

`TrainConfig.terrain` resolves a catalog XML via `scene_path`; workers and the
control room load the same file. Missing fields in old configs default to `flat`.
`build_terrain_assets.py` generates deterministic ramps, steps, rough tiles and
mixed variants for Husky/Go2. These are regular collision geoms, not overlays.

`rl/playback.py` owns display-only sampling and the episode buffer. MuJoCo's
`mj_differentiatePos` / `mj_integratePos` interpolate configuration space including
quaternions, without mutating traces or worker states. The dashboard publishes at
30 Hz. Each worker tile finishes its current episode before loading the newest
completed trace at 1×; no interpolation crosses an episode reset. Raw latest-state
monitoring remains available separately and intentionally follows training speed.
Interpolation is visual reconstruction, not an additional physics rollout; contact
forces computed at intermediate display states are not recorded force measurements.


The control room exposes scene selection and playback controls first; View,
Training and History are collapsed. Scene selection resolves the latest run of
that task, so ALOHA and Go2 never reuse each other's model or traces. The display
clock supports pause/speed without altering worker time. A completed episode is
preferred to an interrupted trailing fragment; optional repeat restarts at the
recorded reset state, never blends across the boundary. ALOHA overview tile spacing,
camera bounds, per-arm trails and target markers are task-aware.

Terrain mesh faces have positive signed volume (outward winding). Small gaps avoid
coplanar sides on adjacent rough tiles. RL hides maze walls only in the viewer;
physical collision geoms are retained. Navigation uses opaque walls and a sparse
metric grid to avoid overlapping translucent faces and visual moire.
