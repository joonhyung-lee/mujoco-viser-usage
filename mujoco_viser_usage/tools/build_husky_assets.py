"""Adapt the local Husky MJCF and URDF inertial data; bundle only used meshes."""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "assets/navigation/husky"


def build(source: Path):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "meshes").mkdir(exist_ok=True)
    original = source / "husky_wo_plate.xml"
    urdf_path = source / "husky.urdf"
    if source.resolve() == ROOT.resolve():
        original, urdf_path = ROOT / "upstream.xml", ROOT / "upstream.urdf"
    else:
        shutil.copy2(original, ROOT / "upstream.xml")
        shutil.copy2(urdf_path, ROOT / "upstream.urdf")
    tree = ET.parse(original)
    root = tree.getroot()
    for mesh in root.findall("asset/mesh"):
        if source.resolve() != ROOT.resolve():
            shutil.copy2(
                source / "meshes" / mesh.get("file"), ROOT / "meshes" / mesh.get("file")
            )
    ET.SubElement(root, "option", timestep=".001", integrator="implicitfast")
    base = root.find("worldbody/body")
    base.set("name", "base")  # Shared navigation/trajectory interface.
    base.set("pos", "0 0 .004")
    inertial = base.find("inertial")
    inertial.attrib.clear()
    urdf = ET.parse(urdf_path).getroot().find("link[@name='base_link']/inertial")
    inertia = urdf.find("inertia")
    center = [float(v) for v in urdf.find("origin").get("xyz").split()]
    center[2] += 0.14493  # URDF base_link -> MJCF base_footprint frame.
    # Upstream MJCF uses a wheel inertia for a 100kg chassis. Restore URDF values.
    inertial.attrib.update(
        pos=" ".join(map(str, center)),
        mass=urdf.find("mass").get("value"),
        fullinertia=" ".join(
            inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    )
    for geom in base.findall("geom"):
        geom.set("conaffinity", "0")
    ET.SubElement(
        base,
        "geom",
        name="chassis_collision",
        type="box",
        size=".5037 .28545 .13375",
        pos="0 0 .26991",
        group="3",
        contype="1",
        conaffinity="1",
        density="0",
    )
    for wheel in base.findall("body"):
        wheel.set("euler", "0 0 3.141592653589793")
        wheel.find("geom").set("conaffinity", "0")
        ET.SubElement(
            wheel,
            "geom",
            name=wheel.get("name") + "_collision",
            type="cylinder",
            size=".17775 .05715",
            quat=".7071067811865476 .7071067811865476 0 0",
            group="3",
            contype="1",
            conaffinity="1",
            density="0",
            friction=".8 .005 .0001",
        )
    for collision in base.iter("geom"):
        if collision.get("group") == "3":
            # Tire compliance takes precedence over the rigid maze floor.
            collision.set("priority", "3")
            collision.set("solref", ".01 1")
            collision.set("solimp", ".95 .99 .001")
    ET.SubElement(base, "site", name="tracking_site", pos="0 0 .48", size=".008")
    ET.SubElement(
        base,
        "camera",
        name="ego_rgb",
        pos=".43 0 .48",
        xyaxes="0 -1 0 0 0 1",
        fovy="75",
    )
    for actuator in root.find("actuator"):
        actuator.set("ctrlrange", "-12 12")
        actuator.set("forcerange", "-25 25")
    ET.indent(root)
    tree.write(ROOT / "husky.xml", encoding="unicode")
    # Retain the shared physical maze while replacing its primitive robot.
    scene = ET.parse(ROOT / "maze.xml")
    maze = scene.getroot()
    for element in list(maze):
        if element.tag in {"compiler", "option", "actuator", "include"}:
            maze.remove(element)
    for body in list(maze.find("worldbody").findall("body")):
        maze.find("worldbody").remove(body)
    maze.insert(0, ET.Element("include", file="husky.xml"))
    maze.set("model", "husky_maze")
    ET.indent(maze)
    scene.write(ROOT / "maze.xml", encoding="unicode")


if __name__ == "__main__":
    build(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("../mujoco-robotics-usage/asset/husky")
    )
