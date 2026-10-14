# Sources and attribution

- The project layout and robotics use cases are inspired by
  https://github.com/joonhyung-lee/mujoco-robotics-usage . Its parser and
  notebook code have not been copied into this project.
- `assets/manipulation/ur5e/ur5e.xml` and the arm `.obj` meshes in `assets/` are copied from
  Google DeepMind MuJoCo Menagerie, `universal_robots_ur5e`, local checkout
  commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.
  https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e
  The original BSD-3-Clause license is retained beside the model.
  `scene.xml` is a new scene containing that model and a falling box.
- Viser documentation: https://viser.studio/main/
- MuJoCo documentation: https://mujoco.readthedocs.io/en/stable/
- Architectural reference: https://github.com/mujocolab/mjviser at commit
  `ae0b219a2049a95e1321214e4efd1d395ee4bf39` (Apache-2.0).
  Referenced loop/scene separation and UI organization; no source code copied.
- Design reference: https://refactoring.guru/design-patterns/what-is-pattern

## RG2 addition

`scenes/manipulation/ur5e/ur5e_rg2.xml` combines the BSD-licensed Menagerie arm above with
RG2 body/inertia/linkage descriptions adapted from the user's local
`mujoco-robotics-usage/asset/ur5e/ur5e_rg2.xml`.
The seven `scenes/manipulation/ur5e/assets/gripper*.STL` files are copied from that repository's
`asset/ur5e/mesh/` directory, not from Menagerie. The reference checkout does not
include a separate license for these RG2 files; they are not covered by the arm's
BSD license merely by being stored beside it.

Changes: omitted camera assembly, raised robot base, added grasp site, adjusted
RG2 damping/armature, actuator force limit, collision masks and linkage constraint
parameters for this contact-based demonstration. Tables and target marker in
`pick_place.xml` are newly authored primitive geometry inspired by the reference
base-table/object-table scene arrangement.

## Domain scene expansion

The following model directories and assets are copied from MuJoCo Menagerie
commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. Each model's LICENSE is retained:

| Bundled directory | Upstream directory | License |
| --- | --- | --- |
| `scenes/manipulation/franka` | `franka_emika_panda` | Apache-2.0 |
| `scenes/manipulation/aloha` | `aloha` | BSD-3-Clause |
| `scenes/locomotion/go1` | `unitree_go1` | BSD-3-Clause |
| `scenes/locomotion/go2` | `unitree_go2` | BSD-3-Clause |
| `scenes/locomotion/spot` | `boston_dynamics_spot` | BSD-3-Clause |

Original upstream XMLs remain alongside our generated variants. Changes in
`task_robot.xml` / `task_keyframe.xml`: append free-object reset poses, name the
ALOHA neutral key `home`. Changes in `*_tracked.xml`: add a base tracking site
and set initial root height from foot geometry. Go2's `go2_position.xml` replaces
torque motors with limited position servos, retaining original torque limits.
Maze floor contacts are firmer than the upstream soft feet; Spot runs at 1 ms.
The new workspace/maze XMLs are authored here. TurtleBot3 remains an original
primitive approximation with simplified mass, shape, and drive parameters.

Architectural references inspected locally; no code, models, maps or datasets
copied from these two benchmarks:

- https://github.com/seohongpark/ocbench at
  `ab6a2c2602b9ac2196e786aea7b7fab2fe779d98`:
  `ocbench/envs/manipulation_env.py`, `ocbench/controllers/diff_ik.py`.
  Referenced separation of environment, physics/control timing, and IK controller.
- https://github.com/seohongpark/ogbench at
  `1d4140997f60c52c6fb0702ec100dc988b18c548`:
  `ogbench/locomaze/maze.py`. Referenced robot-independent grid maze composition.


## Husky MJCF adaptation

`scenes/navigation/husky/upstream.xml`, `upstream.urdf`, and the five referenced
STL files in `meshes/` come from the local `mujoco-robotics-usage/asset/husky`
checkout (`husky_wo_plate.xml`, `husky.urdf`). No separate license for these Husky
files was found in that asset directory; they are not covered by Menagerie licenses.

`husky.xml` preserves the source visual meshes, wheel arrangement and negative
actuator gear. Adaptations: base body alias `base`, 4 mm initial clearance,
URDF chassis mass/center/full inertia replacing the inconsistent MJCF wheel-like
inertia, exact pi wheel orientation, cylinder wheel contacts (source dimensions),
URDF chassis collision size translated into the MJCF visual frame, bounded wheel
controls/forces, compliant tire contacts, 1 ms implicit-fast integration, tracking site and ego camera.
`build_husky_assets.py` reproduces the adapted model from the retained sources.
The terrain variants are newly authored here, deterministic and physically collidable.
