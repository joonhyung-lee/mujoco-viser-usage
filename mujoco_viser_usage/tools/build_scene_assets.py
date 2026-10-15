"""Rebuild bundled scene assets from a local MuJoCo Menagerie checkout.

Usage: uv run python -m mujoco_viser_usage.tools.build_scene_assets ../mujoco_menagerie
Only this development script needs the source checkout; runtime is self-contained.
"""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mujoco_viser_usage.scenes.maze import CELL_SIZE, LAYOUT, world

ROOT = Path(__file__).resolve().parents[2] / "assets"
SOURCE = Path(sys.argv[1])


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def save(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def bundle(category, name, source):
    target = ROOT / category / name
    target.mkdir(parents=True, exist_ok=True)
    for path in (SOURCE / source).iterdir():
        if path.name == "assets":
            shutil.copytree(path, target / "assets", dirs_exist_ok=True)
        elif path.suffix == ".xml" or path.name == "LICENSE":
            shutil.copy2(path, target / path.name)
    return target


def environment(root, maze=False, floor=0):
    # Robot XML already defines integration and actuator defaults.
    add(
        root,
        "statistic",
        center="4 4 0" if maze else "0.25 0 0.3",
        extent="14" if maze else "2",
    )
    visual = add(root, "visual")
    add(visual, "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6")
    wb = add(root, "worldbody")
    add(wb, "light", pos="0 0 8", dir="0 0 -1", directional="true")
    add(
        wb,
        "geom",
        name="floor",
        type="plane",
        size="15 15 .1",
        pos=f"0 0 {floor}",
        rgba=".22 .24 .27 1",
        friction=".9 .01 .001",
        priority="2",
        solref=".004 1",
        solimp=".99 .99 .001",
    )
    if maze:
        for r, line in enumerate(LAYOUT):
            for c, value in enumerate(line):
                x, y = world((r, c))
                if value == "#":
                    add(
                        wb,
                        "geom",
                        name=f"wall_{r}_{c}",
                        type="box",
                        pos=f"{x} {y} .3",
                        size=f"{CELL_SIZE / 2} {CELL_SIZE / 2} .3",
                        rgba=".46 .49 .54 .75",
                    )
                elif value in "SG":
                    add(
                        wb,
                        "site",
                        name="start" if value == "S" else "goal",
                        type="cylinder",
                        pos=f"{x} {y} .008",
                        size=".32 .008",
                        rgba=".3 .8 .55 .65" if value == "G" else ".35 .6 1 .65",
                    )
    for site in wb.findall("site"):
        add(
            wb,
            "geom",
            name=site.get("name") + "_marker",
            type=site.get("type"),
            pos=site.get("pos"),
            size=site.get("size"),
            rgba=site.get("rgba"),
            contype="0",
            conaffinity="0",
        )
    return wb


for name, source, robot in (
    ("go1", "unitree_go1", "go1.xml"),
    ("go2", "unitree_go2", "go2.xml"),
    ("spot", "boston_dynamics_spot", "spot.xml"),
):
    dest = bundle("locomotion", name, source)
    variants = [(name, robot)] + (
        [("spot_arm", "spot_arm.xml")] if name == "spot" else []
    )
    for variant, filename in variants:
        root = ET.Element("mujoco", model=variant + "_maze")
        add(root, "include", file=filename)
        worldbody = environment(root, maze=True)
        if name == "spot":
            # Heavier position-servo model: resolve firm foot contacts at 1 kHz.
            add(root, "option", timestep=".001")
            worldbody.find("geom[@name='floor']").set("solimp", ".999 .999 .001")
        # Go2 upstream uses torque motors, while its home controls are joint angles.
        # Supply a scene-specific position-servo variant so manual controls hold pose.
        if name == "go2":
            model = ET.parse(dest / filename).getroot()
            compiled = mujoco.MjModel.from_xml_path(str((dest / filename).resolve()))
            for i, motor in enumerate(model.find("actuator")):
                motor.tag = "position"
                motor.set("kp", "40")
                motor.set("kv", "2")
                motor.set("ctrllimited", "true")
                motor.set("forcelimited", "true")
                motor.set(
                    "forcerange", " ".join(map(str, compiled.actuator_ctrlrange[i]))
                )
                jid = compiled.actuator_trnid[i, 0]
                motor.set("ctrlrange", " ".join(map(str, compiled.jnt_range[jid])))
            save(model, dest / "go2_position.xml")
            root.find("include").set("file", "go2_position.xml")
        # Track the floating body with the same overlay used by manipulation tools.
        tree = ET.parse(dest / ("go2_position.xml" if name == "go2" else filename))
        home = tree.getroot().find("keyframe/key")
        qpos = home.get("qpos").split()
        source_model = mujoco.MjModel.from_xml_path(str((dest / filename).resolve()))
        source_data = mujoco.MjData(source_model)
        mujoco.mj_resetDataKeyframe(source_model, source_data, 0)
        mujoco.mj_forward(source_model, source_data)
        lowest_foot = min(
            source_data.geom_xpos[i, 2] - source_model.geom_size[i, 0]
            for i in range(source_model.ngeom)
            if source_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
            and source_model.geom_contype[i]
        )
        qpos[2] = str(float(qpos[2]) + 0.001 - lowest_foot)
        home.set("qpos", " ".join(qpos))
        add(
            tree.getroot().find("worldbody/body"),
            "site",
            name="tracking_site",
            pos="0 0 .15",
            size=".008",
        )
        save(tree.getroot(), dest / (variant + "_tracked.xml"))
        root.find("include").set("file", variant + "_tracked.xml")
        save(root, dest / (variant + "_maze.xml"))

for name, source, robot in (
    ("franka", "franka_emika_panda", "panda.xml"),
    ("aloha", "aloha", "aloha.xml"),
):
    dest = bundle("manipulation", name, source)
    # Keep upstream originals. Extend a task copy of the home key for free objects.
    robot_root = ET.parse(dest / robot).getroot()
    key_root = (
        ET.parse(dest / "keyframe_ctrl.xml").getroot()
        if name == "aloha"
        else robot_root
    )
    keys = key_root.find("keyframe")
    key = keys.find("key")
    key.set("name", "home")
    objects = (
        [(0, -0.12, 0.035), (0, 0.12, 0.035)]
        if name == "aloha"
        else [(0.48, -0.16, 0.035), (0.48, 0.16, 0.035)]
    )
    key.set(
        "qpos",
        key.get("qpos") + "".join(f" {x} {y} {z} 1 0 0 0" for x, y, z in objects),
    )
    if name == "aloha":
        save(key_root, dest / "task_keyframe.xml")
        for include in robot_root.findall("include"):
            if include.get("file") == "keyframe_ctrl.xml":
                include.set("file", "task_keyframe.xml")
    save(robot_root, dest / "task_robot.xml")
    root = ET.Element("mujoco", model=name + "_workspace")
    add(root, "include", file="task_robot.xml")
    wb = environment(root, floor=-0.75)
    # ALOHA arms sit at x = +-0.469 on the upstream 1.22 x 0.74 m table, centered.
    # Franka stands at the origin, so its table extends forward.
    aloha = name == "aloha"
    add(
        wb,
        "geom",
        name="object_table",
        type="box",
        size=".61 .37 .045" if aloha else ".72 .55 .045",
        pos="0 0 -.045" if aloha else ".15 0 -.045",
        rgba=".55 .58 .63 1",
    )
    for x in (-0.55, 0.55) if aloha else (-0.45, 0.75):
        for y in (-0.32, 0.32) if aloha else (-0.43, 0.43):
            add(
                wb,
                "geom",
                type="box",
                size=".035 .035 .35",
                pos=f"{x} {y} -.4",
                rgba=".18 .2 .23 1",
            )
    for i, (x, y, z) in enumerate(objects):
        obj = add(wb, "body", name=f"object_{i}", pos=f"{x} {y} {z}")
        add(obj, "freejoint")
        add(
            obj,
            "geom",
            type="box",
            size=".025 .025 .03",
            mass=".05",
            rgba=".95 .97 1 .72",
            friction="1 .01 .001",
        )
    targets = (
        [("left_target", "-.14 .23 .002"), ("right_target", ".14 .23 .002")]
        if name == "aloha"
        else [("target", ".4 .3 .002")]
    )
    for target_name, position in targets:
        add(
            wb,
            "site",
            name=target_name,
            type="cylinder",
            pos=position,
            size=".075 .002",
            rgba=".3 .8 .55 .55",
        )
        add(
            wb,
            "geom",
            name=target_name + "_marker",
            type="cylinder",
            pos=position,
            size=".075 .002",
            rgba=".3 .8 .55 .55",
            contype="0",
            conaffinity="0",
        )
    save(root, dest / "workspace.xml")

# Original primitive approximations: deliberately not presented as vendor CAD.
for name, large in [("husky", True), ("turtlebot3", False)]:
    dest = ROOT / "navigation" / name
    dest.mkdir(parents=True, exist_ok=True)
    radius, halftrack = (0.165, 0.29) if large else (0.065, 0.16)
    root = ET.Element("mujoco", model=name + "_simplified_maze")
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep=".002", integrator="implicitfast")
    wb = environment(root, maze=True)
    base = add(wb, "body", name="base", pos=f"0 0 {radius + 0.008}")
    add(base, "freejoint", name="root")
    add(
        base,
        "geom",
        name="chassis",
        type="box",
        size=".40 .22 .12" if large else ".15 .12 .045",
        pos="0 0 .12",
        mass="25" if large else "3",
        rgba=".72 .65 .18 1" if large else ".24 .26 .3 1",
    )
    add(
        base,
        "geom",
        type="cylinder",
        size=".065 .035",
        pos="0 0 .29",
        mass=".15",
        rgba=".12 .14 .16 1",
    )
    add(base, "site", name="tracking_site", pos="0 0 .32", size=".008")
    add(
        base, "camera", name="ego_rgb", pos=".17 0 .3", xyaxes="0 -1 0 0 0 1", fovy="75"
    )
    actuators = add(root, "actuator")
    # Four-wheel skid steer / differential drive with low-friction caster spheres.
    axles = (-0.25, 0.25) if large else (0,)
    for a, x in enumerate(axles):
        for side, sign in [("left", 1), ("right", -1)]:
            wheel = add(
                base, "body", name=f"{side}_{a}", pos=f"{x} {sign * halftrack} 0"
            )
            add(
                wheel,
                "joint",
                name=f"{side}_{a}",
                type="hinge",
                axis="0 1 0",
                damping=".05",
            )
            add(
                wheel,
                "geom",
                type="cylinder",
                size=f"{radius} .035",
                quat=".70710678 .70710678 0 0",
                mass="1" if large else ".15",
                rgba=".08 .09 .11 1",
                friction=".8 .005 .0001",
                condim="3",
            )
            add(
                actuators,
                "velocity",
                name=f"{side}_{a}",
                joint=f"{side}_{a}",
                kv="12" if large else "1",
                ctrlrange="-12 12",
                forcerange="-25 25" if large else "-2 2",
            )
    if not large:
        for x in (-0.12, 0.12):
            caster = add(base, "body", name=f"caster_{x}", pos=f"{x} 0 -.042")
            add(caster, "joint", type="ball", damping=".00001")
            add(
                caster,
                "geom",
                type="sphere",
                size=".02",
                mass=".02",
                friction=".3 .00001 .00001",
                priority="3",
                solref=".004 1",
                solimp=".99 .99 .001",
            )
    save(root, dest / "maze.xml")

# Replace the temporary maze robot with the bundled source MJCF adaptation.
from build_husky_assets import build as build_husky

build_husky(ROOT / "navigation/husky")
from build_terrain_assets import build as build_terrain

build_terrain()
