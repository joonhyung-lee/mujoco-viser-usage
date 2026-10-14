# mujoco-viser-usage

MuJoCo robots in the browser. Interactive scenes, contact-driven demos, a PPO control
room, and a native [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp) backend,
all rendered with [Viser](https://github.com/nerfstudio-project/viser).

**[Project page](https://joonhyung-lee.github.io/mujoco-viser-usage/)** ·
[Architecture](docs/architecture.md) · [mjlab backend](docs/mjlab.md) · [RL](rl/README.md)

<p align="center">
  <img src="docs/assets/pick-place.png" width="100%" alt="UR5e pick & place with ego RGB, contact forces, and tracked trajectory">
</p>

## Quick start

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run mujoco-viser-usage                          # UR5e + RG2 pick & place → http://127.0.0.1:8080
uv run mujoco-viser-usage --list                   # all scenes
uv run mujoco-viser-usage manipulation/aloha --demo
uv run mujoco-viser-usage navigation/husky_mixed --demo
```

```python
from viser_robotics import ViserParser

with ViserParser(viewer=False) as robot:
    robot.set_control("shoulder_pan", -1.3)
    robot.step(nstep=500)
    q = robot.get_q()
```

## Scenes

| Domain | Robots | Demo |
| --- | --- | --- |
| Manipulation | UR5e + RG2, Franka, ALOHA | Contact-based pick & place (no welds, no teleport) |
| Locomotion | Go1, Go2, Spot, Spot + arm | Go2 analytic trot through position servos |
| Navigation | Husky (CAD meshes), TurtleBot3 | BFS path + wheel-velocity following in a maze |

Go2 and Husky also ship `ramps`, `steps`, `rough`, and `mixed` contact terrains.
Overlays (toggle in **View**): contact-force arrows, wrist-camera RGB, tracked EEF trajectory.

<table>
  <tr>
    <td><img src="docs/assets/aloha.png" alt="ALOHA bimanual pick & place"></td>
    <td><img src="docs/assets/husky-maze.png" alt="Husky maze navigation on mixed terrain"></td>
  </tr>
  <tr>
    <td align="center">ALOHA · two-arm pick & place</td>
    <td align="center">Husky · maze on mixed terrain, ego camera</td>
  </tr>
</table>

## RL control room

```bash
uv sync --extra rl
uv run --extra rl mujoco-viser-rl --port 8090
```

- Tasks: `go2_velocity` (residual over trot), `aloha_reach`, `aloha_pick_place`
- PPO with parallel MuJoCo workers; pause / stop / resume from checkpoints
- Buffered 1× rollout replay, up to four workers side by side, full episode history

<table>
  <tr>
    <td><img src="docs/assets/rl-dashboard.png" alt="RL dashboard replaying an ALOHA reach policy"></td>
    <td><img src="docs/assets/rl-go2.png" alt="RL dashboard replaying a Go2 velocity policy"></td>
  </tr>
  <tr>
    <td align="center">ALOHA reach · mjlab / RSL-RL PPO</td>
    <td align="center">Go2 velocity · SB3 PPO</td>
  </tr>
</table>

## mjlab backend

Same tasks on MuJoCo-Warp with RSL-RL. Isolated environment (mjlab pins MuJoCo 3.11).

```bash
uv sync --project integrations/mjlab
uv run --project integrations/mjlab python -m viser_robotics.mjlab train \
  Viser-Go2-Velocity-v0 --gpu-ids '[0]' --env.scene.num-envs 512
uv run --project integrations/mjlab python -m viser_robotics.mjlab.evaluate <run-dir>
uv run python -m viser_robotics.rl.comparison rl/runs/<comparison-dir>
```

Native tasks: `Viser-Aloha-{Reach,Lift,Place,Bimanual}-v0`, `Viser-Go2-Velocity-v0`.
Evaluation replays scripted control and the policy on the same seed with synchronized
timelines, EEF trails, recorded contact forces, and reconstructed wrist RGB.

<table>
  <tr>
    <td><img src="docs/assets/mjlab-comparison.png" alt="Same-seed ALOHA comparison with replay RGB"></td>
    <td><img src="docs/assets/go2-comparison.png" alt="Same-seed Go2 comparison"></td>
  </tr>
  <tr>
    <td align="center">ALOHA · scripted vs policy, replay RGB</td>
    <td align="center">Go2 · scripted trot vs policy</td>
  </tr>
</table>

## Layout

```text
src/viser_robotics/
  viser_parser.py   public entry point        scenes/{manipulation,locomotion,navigation}/
  simulation.py     model, control, physics   rl/        SB3 PPO, dashboard, comparison viewer
  viewer.py         Viser scene + camera      mjlab/     native Warp tasks, recorder, evaluate
  overlays.py       forces, RGB, trajectory   integrations/mjlab/   pinned mjlab environment
```

## Validate

```bash
uv run pytest -q
uv run ruff check src tests examples scripts
uv run --project integrations/mjlab --with pytest python -m pytest tests/test_mjlab.py -q
```

Robot models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
and related sources; see [THIRD_PARTY.md](THIRD_PARTY.md).
