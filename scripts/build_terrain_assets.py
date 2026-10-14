"""Deterministic collision terrain variants, shared by RL and interactive scenes."""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/viser_robotics/scenes"


def obstacle(wb, name, x, y, z, sx, sy, sz, color=".70 .76 .78 1"):
    ET.SubElement(
        wb,
        "geom",
        name="terrain_" + name,
        type="box",
        pos=f"{x} {y} {z}",
        size=f"{sx} {sy} {sz}",
        rgba=color,
        friction=".9 .01 .001",
        priority="2",
        solref=".004 1",
        solimp=".99 .99 .001",
    )


def ramp(root, wb, x):
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    # A triangular prism produces a continuous up/down ramp without a ledge.
    ET.SubElement(
        assets,
        "mesh",
        name="terrain_ramp",
        vertex="-.6 -.7 0  .6 -.7 0  0 -.7 .08  -.6 .7 0  .6 .7 0  0 .7 .08",
        face="0 1 2 3 5 4 0 4 1 0 3 4 1 5 2 1 4 5 2 3 0 2 5 3",
    )
    ET.SubElement(
        wb,
        "geom",
        name="terrain_ramp",
        type="mesh",
        mesh="terrain_ramp",
        pos=f"{x} 0 0",
        rgba=".70 .74 .76 1",
        friction=".9 .01 .001",
        priority="2",
        solref=".004 1",
        solimp=".99 .99 .001",
    )


def build():
    # Brighten MuJoCo RGB as well as the independent Viser renderer.
    for path in ROOT.rglob("*.xml"):
        if path.name not in {
            "maze.xml",
            "workspace.xml",
            "go1_maze.xml",
            "go2_maze.xml",
            "spot_maze.xml",
            "spot_arm_maze.xml",
        }:
            continue
        tree = ET.parse(path)
        floor = tree.find("worldbody/geom[@name='floor']")
        if floor is not None:
            floor.set("rgba", ".72 .76 .80 1")
        headlight = tree.find("visual/headlight")
        if headlight is not None:
            headlight.set("ambient", ".5 .5 .5")
            headlight.set("diffuse", ".7 .7 .7")
        for wall in tree.findall("worldbody/geom"):
            if wall.get("name", "").startswith("wall_"):
                wall.set("rgba", ".68 .72 .76 1")
        ET.indent(tree)
        tree.write(path, encoding="unicode")
    for relative in ("locomotion/go2/go2_maze.xml", "navigation/husky/maze.xml"):
        source = ROOT / relative
        for variant in ("ramps", "steps", "rough", "mixed"):
            tree = ET.parse(source)
            root = tree.getroot()
            root.set("model", source.parent.name + "_" + variant)
            wb = root.find("worldbody")
            if variant in {"ramps", "mixed"}:
                ramp(root, wb, 1.65)
            if variant in {"steps", "mixed"}:
                center = 2.65 if variant == "mixed" else 1.8
                for i, height in enumerate((0.015, 0.03, 0.045, 0.03, 0.015)):
                    obstacle(
                        wb,
                        f"step_{i}",
                        center + (i - 2) * 0.16,
                        0,
                        height / 2,
                        0.08,
                        0.7,
                        height / 2,
                        ".72 .75 .77 1",
                    )
            if variant in {"rough", "mixed"}:
                start = 4.0 if variant == "mixed" else 1.2
                for i in range(8):
                    for j in range(5):
                        height = 0.008 + 0.004 * ((i * 7 + j * 3) % 5)
                        obstacle(
                            wb,
                            f"rough_{i}_{j}",
                            start + i * 0.19,
                            (j - 2) * 0.25,
                            height / 2,
                            0.094,
                            0.124,
                            height / 2,
                            ".69 .73 .75 1",
                        )
            ET.indent(tree)
            tree.write(source.parent / f"{variant}.xml", encoding="unicode")


if __name__ == "__main__":
    build()
