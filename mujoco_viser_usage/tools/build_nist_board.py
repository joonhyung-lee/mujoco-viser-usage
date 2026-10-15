"""Build the NIST Assembly Task Board #1 scene from the public NIST STL kit.

Usage:
  uv run --with fast-simplification python -m mujoco_viser_usage.tools.build_nist_board <taskboard_1_stl dir>

STL kit: https://www.nist.gov/document/taskboard1stlzip (public domain, NIST).
Hole positions come from a mid-plane section of GMC_Laser_Plate_Virtual.STL.
MuJoCo collides meshes as convex hulls, so every hole, bore, and nut is built
from primitive rings; the decimated STLs are visual only. Units: NIST mm -> m.
"""

import itertools
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2] / "assets/manipulation/franka"
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else None

BOARD = 384.0  # mm square
THICK = 9.0  # 0.354 in acrylic
ORIGIN = np.array(
    [0.52, 0.0, 0.0]
)  # board center on the Franka table; near edge 0.33 m from the base

# Plate frame (mm). Round holes drilled per the NIST instructions.
ROUND = {
    4: ((191.4, 347.2), 4.1),
    8: ((116.4, 347.2), 8.1),
    12: ((41.4, 122.2), 12.2),
    16: ((266.4, 122.2), 16.5),
}
SQUARE = {  # name: center, peg (x, y), laser-cut hole (x, y)
    4: ((191.4, 197.2), (4.0, 4.0), (4.1, 4.1)),
    8: ((266.4, 272.2), (8.0, 7.0), (8.2, 7.1)),
    12: ((41.4, 197.2), (8.0, 12.0), (8.1, 12.2)),
    16: ((41.4, 347.2), (16.0, 10.0), (16.3, 10.2)),
}
BOLTS = {
    4: ((191.4, 122.2), 10.0, 7.0, 3.2),
    8: ((41.4, 47.2), 12.0, 13.0, 6.5),
    12: ((266.4, 347.2), 14.0, 19.0, 10.0),
    16: ((341.4, 47.2), 20.0, 24.0, 13.0),
}
GEAR_PLATE = (181.4, 47.2)
GEARS = {
    "large": (141.4, 31.0, 62.0),
    "medium": (191.4, 21.0, 42.0),
    "small": (221.4, 11.0, 22.0),
}
CONNECTORS = {
    "dsub": (116.4, 197.2),
    "usb": (266.4, 197.2),
    "rj45": (41.4, 272.2),
    "bnc": (341.4, 347.2),
    "waterproof": (191.4, 279.7),
}
RING = 2.0  # wall thickness of primitive hole rings, mm
LEAD = 3.0  # 45 degree lead-in at every hole mouth (NIST chamfer option), mm
TRAY_Y = -0.30  # kit tray row on the table, world m

FIRM = {"priority": "2", "solref": ".004 1", "solimp": ".99 .99 .001"}
MESHES = {
    "plate": "GMC_Laser_Plate_Virtual.STL",
    "gear_plate": "Gear_Plate.STL",
    "gear_shaft": "Gear_Shaft.STL",
    "gear_large": "Gear_Large.STL",
    "gear_medium": "Gear_Medium.STL",
    "gear_small": "Gear_Small.STL",
    "nut_4": "M4_Hex_Nut.STL",
    "nut_8": "M8_Hex_Nut.STL",
    "nut_12": "M12_Hex_Nut.STL",
    "nut_16": "M16_Hex_Nut.STL",
    "dsub_f": "DSUB_Female.STL",
    "dsub_housing": "DB_Housing.STL",
    "usb_f": "USB_Female.STL",
    "rj45_f": "RJ45_Female.STL",
    "rj45_housing": "RJ45_Housing.STL",
    "rj45_m": "RJ45_Male.STL",
    "bnc_f": "BNC_Female.STL",
    "bnc_m": "BNC_Male.STL",
    "waterproof_f": "Waterproof_Female.STL",
    "waterproof_m": "Waterproof_Male.STL",
}


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def fmt(*values):
    return " ".join(f"{v:.6g}" for v in values)


def world(px, py, z=0.0):
    """Plate frame (mm, corner origin) to world (m)."""
    return ORIGIN + np.array(
        [(px - BOARD / 2) / 1000, (py - BOARD / 2) / 1000, z / 1000]
    )


def prepare_meshes():
    """Decimate, recenter (xy center, z min = 0), binary STL in mm."""
    import fast_simplification

    out = ROOT / "assets/nist"
    out.mkdir(parents=True, exist_ok=True)
    extents = {}
    for name, file in MESHES.items():
        mesh = trimesh.load(SOURCE / file, force="mesh")
        if len(mesh.faces) > 3000 and name != "plate":
            v, f = fast_simplification.simplify(
                mesh.vertices, mesh.faces, target_count=3000
            )
            mesh = trimesh.Trimesh(v, f)
        lo, hi = mesh.bounds
        mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        mesh.export(out / f"{name}.stl")
        extents[name] = mesh.extents
    return extents


def rectangles(windows):
    """Axis-aligned boxes covering the plate minus square windows (plate mm)."""
    xs = sorted({0.0, BOARD} | {x for w in windows for x in (w[0], w[1])})
    boxes = []
    for x0, x1 in itertools.pairwise(xs):
        cuts = sorted(
            {0.0, BOARD}
            | {y for w in windows if w[0] < x1 and w[1] > x0 for y in (w[2], w[3])}
        )
        for y0, y1 in itertools.pairwise(cuts):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(w[0] < cx < w[1] and w[2] < cy < w[3] for w in windows):
                continue
            boxes.append((x0, x1, y0, y1))
    # Merge vertical neighbours sharing an x-span to keep the geom count down.
    merged = []
    for box in sorted(boxes):
        if merged and merged[-1][:2] == box[:2] and abs(merged[-1][3] - box[2]) < 1e-9:
            merged[-1] = (box[0], box[1], merged[-1][2], box[3])
        else:
            merged.append(box)
    return merged


def ring(
    parent, name, center, inner, outer, height, z0, sides, rgba, phase=0.5, **extra
):
    """Polygonal ring of boxes tangent to radius `inner`, in plate mm around world `center` (m)."""
    half = (outer - inner) / 2
    for i in range(sides):
        angle = 2 * np.pi * (i + phase) / sides
        radial = inner + half
        pos = (
            center
            + np.array([np.cos(angle) * radial, np.sin(angle) * radial, 0]) / 1000
        )
        pos[2] = z0 + height / 2000
        add(
            parent,
            "geom",
            name=f"{name}_{i}",
            type="box",
            pos=fmt(*pos),
            size=fmt(
                half / 1000,
                (inner + half) * np.tan(np.pi / sides) / 1000,
                height / 2000,
            ),
            axisangle=fmt(0, 0, 1, angle),
            rgba=rgba,
            **FIRM,
            **extra,
        )


def chamfer(parent, name, center, faces, rgba):
    """45 degree lead-in boxes; faces = (angle, distance of the wall, half width) in mm."""
    t = 1.5
    for i, (angle, dist, half_width) in enumerate(faces):
        radial = dist + LEAD / 2 + t / (2 * np.sqrt(2))
        z = THICK - LEAD / 2 - t / (2 * np.sqrt(2))
        pos = (
            center
            + np.array([np.cos(angle) * radial, np.sin(angle) * radial, 0]) / 1000
        )
        pos[2] = z / 1000
        qz, qy, quat = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qz, np.array([0.0, 0, 1]), angle)
        mujoco.mju_axisAngle2Quat(qy, np.array([0.0, 1, 0]), -np.pi / 4)
        mujoco.mju_mulQuat(quat, qz, qy)
        add(
            parent,
            "geom",
            name=f"{name}_c{i}",
            type="box",
            pos=fmt(*pos),
            size=fmt(LEAD / np.sqrt(2) / 1000, half_width / 1000, t / 2000),
            quat=fmt(*quat),
            rgba=rgba,
            **FIRM,
        )


def box_ring(parent, name, center, inner_xy, wall, height, z0, rgba):
    """Rectangular pocket walls around a square hole; inner_xy are full hole sizes (mm)."""
    hx, hy = inner_xy[0] / 2, inner_xy[1] / 2
    for i, (dx, dy, sx, sy) in enumerate(
        (
            (hx + wall / 2, 0, wall / 2, hy + wall),
            (-hx - wall / 2, 0, wall / 2, hy + wall),
            (0, hy + wall / 2, hx, wall / 2),
            (0, -hy - wall / 2, hx, wall / 2),
        )
    ):
        pos = center + np.array([dx, dy, 0]) / 1000
        pos[2] = z0 + height / 2000
        add(
            parent,
            "geom",
            name=f"{name}_{i}",
            type="box",
            pos=fmt(*pos),
            size=fmt(sx / 1000, sy / 1000, height / 2000),
            rgba=rgba,
            **FIRM,
        )


def build(extents):
    root = ET.Element("mujoco", model="nist_task_board_1")
    add(root, "include", file="franka.xml")
    add(root, "statistic", center="0.5 -0.05 0.05", extent="0.7")
    add(add(root, "visual"), "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6")
    asset = add(root, "asset")
    for name in MESHES:
        add(
            asset, "mesh", name=name, file=f"nist/{name}.stl", scale="0.001 0.001 0.001"
        )
    wb = add(root, "worldbody")
    add(wb, "light", pos="0 0 8", dir="0 0 -1", directional="true")
    add(wb, "light", pos="0.4 -0.6 1.5", dir="0 0.4 -1", diffuse=".4 .4 .4")
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
    add(
        wb,
        "geom",
        name="object_table",
        type="box",
        size=".55 .45 .045",
        pos=".3 0 -.045",
        rgba=".55 .58 .63 1",
        **FIRM,
    )
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

    # Board: visual STL, collision = plate boxes minus windows + hole rings.
    board = add(wb, "body", name="board", pos=fmt(*ORIGIN))
    add(
        board,
        "geom",
        type="mesh",
        mesh="plate",
        pos=fmt(0, 0, 0),
        rgba=".93 .93 .9 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    windows = []
    for size, (center, dia) in ROUND.items():
        r = dia / 2 + RING * 2 + LEAD
        windows.append((center[0] - r, center[0] + r, center[1] - r, center[1] + r))
    for size, (center, _, hole) in SQUARE.items():
        wx, wy = hole[0] / 2 + RING + LEAD, hole[1] / 2 + RING + LEAD
        windows.append((center[0] - wx, center[0] + wx, center[1] - wy, center[1] + wy))
    for i, (x0, x1, y0, y1) in enumerate(rectangles(windows)):
        c = world((x0 + x1) / 2, (y0 + y1) / 2, THICK / 2) - ORIGIN
        add(
            board,
            "geom",
            name=f"plate_{i}",
            type="box",
            pos=fmt(*c),
            size=fmt((x1 - x0) / 2000, (y1 - y0) / 2000, THICK / 2000),
            rgba=".93 .93 .9 0",
            group="3",
            **FIRM,
        )
    for size, (center, dia) in ROUND.items():
        ring(
            board,
            f"hole_r{size}",
            world(*center) - ORIGIN,
            dia / 2,
            dia / 2 + RING + LEAD,
            THICK - LEAD,
            0.0,
            12,
            ".93 .93 .9 0",
            group="3",
        )
        chamfer(
            board,
            f"hole_r{size}",
            world(*center) - ORIGIN,
            [
                (
                    2 * np.pi * (i + 0.5) / 12,
                    dia / 2,
                    (dia / 2 + LEAD + 2) * np.tan(np.pi / 12) + 0.3,
                )
                for i in range(12)
            ],
            ".93 .93 .9 0",
        )
    for size, (center, _, hole) in SQUARE.items():
        box_ring(
            board,
            f"hole_s{size}",
            world(*center) - ORIGIN,
            hole,
            RING + LEAD,
            THICK - LEAD,
            0.0,
            ".93 .93 .9 0",
        )
        hx, hy = hole[0] / 2, hole[1] / 2
        chamfer(
            board,
            f"hole_s{size}",
            world(*center) - ORIGIN,
            [
                (0, hx, hy + LEAD + 2),
                (np.pi, hx, hy + LEAD + 2),
                (np.pi / 2, hy, hx + LEAD + 2),
                (-np.pi / 2, hy, hx + LEAD + 2),
            ],
            ".93 .93 .9 0",
        )

    # Bolts standing in the tapped holes (threads are not simulated).
    for size, (center, shank, flats, nut_h) in BOLTS.items():
        c = world(*center, THICK) - ORIGIN
        add(
            board,
            "geom",
            name=f"bolt_{size}",
            type="capsule",
            pos=fmt(c[0], c[1], c[2] + shank / 2000),
            size=fmt(size / 2000, max(shank / 2 - size / 2, 0.5) / 1000),
            rgba=".55 .55 .6 1",
            **FIRM,
        )
        add(
            board,
            "site",
            name=f"bolt_{size}_seat",
            pos=fmt(c[0], c[1], c[2]),
            size=".002",
            rgba="1 .4 .2 .5",
            group="5",
        )

    # Gear plate with three shafts; gears mesh at pitch radii 31 / 21 / 11.
    gp = world(*GEAR_PLATE, THICK) - ORIGIN
    add(
        board,
        "geom",
        type="mesh",
        mesh="gear_plate",
        pos=fmt(*gp),
        rgba=".8 .8 .82 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    add(
        board,
        "geom",
        name="gear_plate",
        type="box",
        pos=fmt(gp[0], gp[1], gp[2] + 0.0025),
        size=fmt(0.06, 0.03, 0.0025),
        rgba=".8 .8 .82 0",
        group="3",
        **FIRM,
    )
    for name, (x, pitch, outer) in GEARS.items():
        s = world(x, GEAR_PLATE[1], THICK + 5) - ORIGIN
        add(
            board,
            "geom",
            name=f"shaft_{name}",
            type="capsule",
            pos=fmt(s[0], s[1], s[2] + 0.01),
            size=fmt(0.005, 0.005),
            rgba=".6 .6 .65 1",
            **FIRM,
        )
        add(
            board,
            "site",
            name=f"shaft_{name}_seat",
            pos=fmt(*s),
            size=".002",
            rgba="1 .4 .2 .5",
            group="5",
        )

    # Fixed female connectors (visual + hull box); males are loose parts in the tray.
    for name, center in CONNECTORS.items():
        c = world(*center, THICK) - ORIGIN
        for part in (f"{name}_housing", f"{name}_f"):
            if part in extents:
                add(
                    board,
                    "geom",
                    type="mesh",
                    mesh=part,
                    pos=fmt(*c),
                    rgba=".25 .25 .28 1",
                    contype="0",
                    conaffinity="0",
                    group="2",
                )
        ext = extents[f"{name}_f"]
        add(
            board,
            "geom",
            name=f"{name}_f",
            type="box",
            pos=fmt(c[0], c[1], c[2] + ext[2] / 2000),
            size=fmt(ext[0] / 2000, ext[1] / 2000, ext[2] / 2000),
            rgba=".25 .25 .28 0",
            group="3",
            **FIRM,
        )

    # Kit tray: parts start upright in loose sockets beside the board.
    tray = add(wb, "body", name="tray", pos=fmt(0.5, TRAY_Y, 0))
    add(
        tray,
        "geom",
        name="tray_plate",
        type="box",
        pos="0 0 .0015",
        size=".2 .05 .0015",
        rgba=".3 .32 .36 1",
        **FIRM,
    )
    parts = []
    slots = iter(np.linspace(-0.18, 0.18, 10))
    masses = {4: 0.005, 8: 0.02, 12: 0.045, 16: 0.08}
    for size, (center, dia) in ROUND.items():
        x = next(slots)
        ring(
            tray,
            f"tray_r{size}",
            np.array([x, 0, 0]),
            dia / 2 + 1.0,
            dia / 2 + 4.0,
            6.0,
            0.003,
            8,
            ".3 .32 .36 1",
        )
        body = add(
            wb, "body", name=f"rod_{size}", pos=fmt(0.5 + x, TRAY_Y, 0.003 + 0.025)
        )
        add(body, "freejoint", name=f"rod_{size}")
        add(
            body,
            "geom",
            name=f"rod_{size}",
            type="cylinder",
            size=fmt(size / 2000, 0.025),
            mass=masses[size],
            rgba=".75 .76 .8 1",
            condim="6",
            friction="1 .02 .01",
            **FIRM,
        )
        add(
            body,
            "site",
            name=f"rod_{size}_tip",
            pos="0 0 -0.025",
            size=".002",
            group="5",
        )
        parts.append(
            (
                f"rod_{size}",
                (0.5 + x, TRAY_Y, 0.028),
                world(*center, 25.0),
                f"hole_r{size}",
            )
        )
    for size, (center, peg, hole) in SQUARE.items():
        x = next(slots)
        box_ring(
            tray,
            f"tray_s{size}",
            np.array([x, 0, 0]),
            (peg[0] + 2, peg[1] + 2),
            3.0,
            6.0,
            0.003,
            ".3 .32 .36 1",
        )
        body = add(
            wb, "body", name=f"peg_{size}", pos=fmt(0.5 + x, TRAY_Y, 0.003 + 0.025)
        )
        add(body, "freejoint", name=f"peg_{size}")
        add(
            body,
            "geom",
            name=f"peg_{size}",
            type="box",
            size=fmt(peg[0] / 2000, peg[1] / 2000, 0.025),
            mass=masses[size],
            rgba=".75 .76 .8 1",
            condim="6",
            friction="1 .02 .01",
            **FIRM,
        )
        add(
            body,
            "site",
            name=f"peg_{size}_tip",
            pos="0 0 -0.025",
            size=".002",
            group="5",
        )
        parts.append(
            (
                f"peg_{size}",
                (0.5 + x, TRAY_Y, 0.028),
                world(*center, 25.0),
                f"hole_s{size}",
            )
        )
    # Gears and nuts lie flat on the table on the far side of the board.
    for i, (name, (x, pitch, outer)) in enumerate(GEARS.items()):
        pos = (0.30 + 0.09 * i, -TRAY_Y, 0.0)
        body = add(wb, "body", name=f"gear_{name}", pos=fmt(*pos))
        add(body, "freejoint", name=f"gear_{name}")
        add(
            body,
            "geom",
            type="mesh",
            mesh=f"gear_{name}",
            rgba=".2 .2 .22 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ring(
            body,
            f"gear_{name}",
            np.zeros(3),
            5.3,
            pitch - 1.0,
            20.0,
            0.0,
            12,
            ".2 .2 .22 0",
            phase=0.0,
            mass=fmt(0.03 * outer / 42 / 12),
            condim="6",
            friction="1 .02 .01",
        )
        seat = world(x, GEAR_PLATE[1], THICK + 5)
        parts.append((f"gear_{name}", pos, seat, f"shaft_{name}"))
    for i, (size, (center, shank, flats, nut_h)) in enumerate(BOLTS.items()):
        pos = (0.58 + 0.04 * i, -TRAY_Y, 0.0)
        body = add(wb, "body", name=f"nut_{size}", pos=fmt(*pos))
        add(body, "freejoint", name=f"nut_{size}")
        add(
            body,
            "geom",
            type="mesh",
            mesh=f"nut_{size}",
            rgba=".6 .6 .65 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ring(
            body,
            f"nut_{size}",
            np.zeros(3),
            size / 2 + 0.5,
            flats / 2,
            nut_h,
            0.0,
            6,
            ".6 .6 .65 0",
            mass=fmt(0.0003 * flats * flats * nut_h / 6 / 100),
            condim="6",
            friction="1 .02 .01",
        )
        parts.append((f"nut_{size}", pos, world(*center, THICK), f"bolt_{size}"))
    for name in ("dsub", "usb", "rj45", "bnc", "waterproof"):
        if f"{name}_m" not in extents:
            continue
        ext = extents[f"{name}_m"]
        pos = (
            0.25 + 0.1 * len([p for p in parts if p[0].endswith("_m")]),
            TRAY_Y - 0.09,
            0.0,
        )
        body = add(wb, "body", name=f"{name}_m", pos=fmt(*pos))
        add(body, "freejoint", name=f"{name}_m")
        add(
            body,
            "geom",
            type="mesh",
            mesh=f"{name}_m",
            rgba=".25 .25 .28 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        add(
            body,
            "geom",
            name=f"{name}_m",
            type="box",
            pos=fmt(0, 0, ext[2] / 2000),
            size=fmt(ext[0] / 2000, ext[1] / 2000, ext[2] / 2000),
            mass="0.02",
            rgba=".25 .25 .28 0",
            **FIRM,
        )
        parts.append((f"{name}_m", pos, None, None))

    # Assembled poses as sites for success checks (AlphaNIST: 1 mm).
    for name, _, seat, _ in parts:
        if seat is not None:
            add(
                wb,
                "site",
                name=f"{name}_goal",
                pos=fmt(*seat),
                size=".002",
                rgba="1 .4 .2 .4",
                group="5",
            )
    key = add(root, "keyframe")
    qpos = ["0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"]
    for _, pos, _, _ in parts:
        qpos.append(f"{fmt(*pos)} 1 0 0 0")
    add(
        key,
        "key",
        name="home",
        qpos=" ".join(qpos),
        ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255",
    )
    return root


def main():
    extents = prepare_meshes()
    root = build(extents)
    ET.indent(root)
    ET.ElementTree(root).write(ROOT / "nist_board.xml", encoding="unicode")
    print(ROOT / "nist_board.xml")


if __name__ == "__main__":
    main()
