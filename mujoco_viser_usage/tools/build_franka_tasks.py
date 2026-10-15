"""Generate the contact-rich Franka task scenes next to the Franka workspace.

Usage: uv run python -m mujoco_viser_usage.tools.build_franka_tasks
Pure geometry; no Menagerie checkout needed.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "assets/manipulation/franka"

TABLE = {"size": ".55 .45 .045", "pos": ".3 0 -.045", "rgba": ".55 .58 .63 1"}
FIRM = {"priority": "2", "solref": ".004 1", "solimp": ".99 .99 .001"}

# Plug insertion: 2.5 mm clearance per side between prong and socket pocket.
PRONG = (0.0085, 0.0165, 0.015)
POCKET = (0.011, 0.019, 0.0125)
SOCKET = (0.5, 0.2)
HOLDER = (0.4, -0.2)

# Cable routing: capsule chain anchored at the kettle, free plug at the far end.
SEGMENTS = 14
SEGMENT = 0.03
ANCHOR = (0.58, 0.3)
CLIP = (0.38, 0.02)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def scene(model):
    root = ET.Element("mujoco", model=model)
    add(root, "include", file="franka.xml")
    add(root, "statistic", center="0.45 0.05 0.05", extent="1.0")
    add(add(root, "visual"), "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6")
    wb = add(root, "worldbody")
    add(wb, "light", pos="0 0 8", dir="0 0 -1", directional="true")
    add(
        wb,
        "geom",
        name="floor",
        type="plane",
        size="15 15 .1",
        pos="0 0 -0.75",
        rgba=".72 .76 .80 1",
        friction=".9 .01 .001",
        **FIRM,
    )
    add(wb, "geom", name="object_table", type="box", **TABLE, **FIRM)
    for x in (-0.2, 0.8):
        for y in (-0.38, 0.38):
            add(
                wb,
                "geom",
                type="box",
                size=".035 .035 .35",
                pos=f"{x} {y} -.4",
                rgba=".18 .2 .23 1",
            )
    return root, wb


def pocket(wb, prefix, center, inner, height, wall, rgba):
    """Four walls around an open-top rectangular pocket on the table."""
    x, y = center
    hx, hy = inner
    for name, pos, size in (
        ("x0", (x - hx - wall, y, height), (wall, hy + 2 * wall, height)),
        ("x1", (x + hx + wall, y, height), (wall, hy + 2 * wall, height)),
        ("y0", (x, y - hy - wall, height), (hx, wall, height)),
        ("y1", (x, y + hy + wall, height), (hx, wall, height)),
    ):
        add(
            wb,
            "geom",
            name=f"{prefix}_{name}",
            type="box",
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            rgba=rgba,
            **FIRM,
        )


def kettle(wb, x, y):
    add(
        wb,
        "geom",
        name="kettle",
        type="cylinder",
        pos=f"{x} {y} 0.08",
        size=".055 .08",
        rgba=".82 .84 .86 1",
    )
    add(
        wb,
        "geom",
        type="box",
        pos=f"{x} {y + 0.07} 0.1",
        size=".012 .02 .03",
        rgba=".2 .2 .22 1",
    )


def plug_scene():
    root, wb = scene("franka_plug")
    kettle(wb, SOCKET[0], SOCKET[1] + 0.11)
    pocket(wb, "socket", SOCKET, POCKET[:2], POCKET[2], 0.01, ".2 .2 .22 1")
    add(
        wb,
        "site",
        name="socket_bottom",
        pos=f"{SOCKET[0]} {SOCKET[1]} 0",
        size=".004",
        rgba="1 .4 .2 .6",
    )
    pocket(wb, "holder", HOLDER, (0.013, 0.021), 0.0125, 0.008, ".45 .47 .5 1")
    handle_z = 2 * PRONG[2] + 0.02
    plug = add(wb, "body", name="plug", pos=f"{HOLDER[0]} {HOLDER[1]} {handle_z}")
    add(plug, "freejoint", name="plug_free")
    add(
        plug,
        "geom",
        name="plug_handle",
        type="box",
        size=".015 .03 .02",
        mass=".06",
        rgba=".95 .97 1 .9",
        friction="1 .01 .001",
        **FIRM,
    )
    add(
        plug,
        "geom",
        name="plug_prong",
        type="box",
        pos=f"0 0 {-0.02 - PRONG[2]}",
        size=" ".join(map(str, PRONG)),
        mass=".02",
        rgba=".75 .6 .2 1",
        friction=".4 .01 .001",
        **FIRM,
    )
    add(plug, "site", name="plug_tip", pos=f"0 0 {-0.02 - 2 * PRONG[2]}", size=".003")
    key = add(root, "keyframe")
    home = "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
    add(
        key,
        "key",
        name="home",
        qpos=f"{home} {HOLDER[0]} {HOLDER[1]} {handle_z} 1 0 0 0",
        ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255",
    )
    return root


def cable_scene():
    root, wb = scene("franka_cable")
    kettle(wb, ANCHOR[0], ANCHOR[1] + 0.08)
    radius = 0.004
    parent = add(
        wb, "body", name="cable_0", pos=f"{ANCHOR[0]} {ANCHOR[1]} {radius + 0.001}"
    )
    add(parent, "joint", name="cable_0", type="ball", damping=".002", stiffness=".02")
    for i in range(SEGMENTS):
        add(
            parent,
            "geom",
            name=f"cable_{i}",
            type="capsule",
            fromto=f"0 0 0 {-SEGMENT} 0 0",
            size=str(radius),
            mass=".004",
            rgba=".15 .15 .17 1",
            friction=".6 .01 .001",
            **FIRM,
        )
        if i == SEGMENTS - 1:
            break
        parent = add(parent, "body", name=f"cable_{i + 1}", pos=f"{-SEGMENT} 0 0")
        add(
            parent,
            "joint",
            name=f"cable_{i + 1}",
            type="ball",
            damping=".002",
            stiffness=".02",
        )
    end = add(parent, "body", name="cable_end", pos=f"{-SEGMENT - 0.012} 0 0.008")
    add(end, "joint", name="cable_end", type="ball", damping=".002", stiffness=".02")
    add(
        end,
        "geom",
        name="cable_plug",
        type="box",
        size=".018 .012 .012",
        mass=".03",
        rgba=".95 .97 1 .9",
        # Torsional friction: chain tension would otherwise spin the plug in the pads.
        condim="6",
        friction="1 .02 .01",
        **FIRM,
    )
    # U-channel open along the cable so chain tension cannot wedge the plug on a wall.
    for name, y in (("y0", CLIP[1] - 0.026), ("y1", CLIP[1] + 0.026)):
        add(
            wb,
            "geom",
            name=f"clip_{name}",
            type="box",
            pos=f"{CLIP[0]} {y} 0.015",
            size=".03 .008 .015",
            rgba=".2 .2 .22 1",
            **FIRM,
        )
    add(
        wb,
        "site",
        name="clip_center",
        pos=f"{CLIP[0]} {CLIP[1]} 0.012",
        size=".004",
        rgba="1 .4 .2 .6",
    )
    key = add(root, "keyframe")
    home = "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
    quats = " ".join("1 0 0 0" for _ in range(SEGMENTS + 1))
    add(
        key,
        "key",
        name="home",
        qpos=f"{home} {quats}",
        ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255",
    )
    return root


def robot():
    """Upstream Panda without its keyframe, plus a TCP site between the fingertips."""
    root = ET.parse(ROOT / "panda.xml").getroot()
    for key in root.findall("keyframe"):
        root.remove(key)
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tcp", pos="0 0 0.1034", size=".004", group="5")
    ET.indent(root)
    ET.ElementTree(root).write(ROOT / "franka.xml", encoding="unicode")


def main():
    robot()
    for name, build in (("plug", plug_scene), ("cable", cable_scene)):
        root = build()
        ET.indent(root)
        ET.ElementTree(root).write(ROOT / f"{name}.xml", encoding="unicode")
        print(ROOT / f"{name}.xml")


if __name__ == "__main__":
    main()
