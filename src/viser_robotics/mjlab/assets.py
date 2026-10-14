"""Reuse bundled ALOHA assets; keep detached objects separate mjlab entities."""

from functools import partial

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..scenes import ROOT


def robot_spec():
    directory = ROOT / "manipulation/aloha"
    assets = {
        path.name: path.read_bytes() for path in (directory / "assets").glob("*.stl")
    }
    spec = mujoco.MjSpec.from_file(str(directory / "aloha.xml"), assets=assets)
    for key in list(spec.keys):
        spec.delete(key)
    for actuator in spec.actuators:
        if "gripper" in actuator.name:
            actuator.forcelimited = True
            actuator.forcerange = [-4, 4]
    return spec


def object_spec(index):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=f"object_{index}")
    body.add_freejoint(name=f"object_{index}_free")
    body.add_geom(
        name=f"object_{index}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.025, 0.025, 0.03),
        mass=0.05,
        rgba=(0.95, 0.97, 1, 0.72),
        friction=(1, 0.01, 0.001),
        priority=2,
        solref=(0.004, 1),
        solimp=(0.99, 0.99, 0.001, 0.5, 2),
    )
    return spec


def copy_world_geoms(spec, source, prefix="world", **overrides):
    """Copy a compiled model's static world geoms (and their meshes) into a spec.

    Static geometry is shared in layout, but each Warp world has independent contacts.
    """
    meshes = set()
    for i in range(source.ngeom):
        if source.geom_bodyid[i] != 0:
            continue
        fields = {
            "type": source.geom_type[i],
            "size": source.geom_size[i],
            "pos": source.geom_pos[i],
            "quat": source.geom_quat[i],
            "rgba": source.geom_rgba[i],
            "contype": int(source.geom_contype[i]),
            "conaffinity": int(source.geom_conaffinity[i]),
            "condim": int(source.geom_condim[i]),
            "friction": source.geom_friction[i],
            "priority": int(source.geom_priority[i]),
            "solref": source.geom_solref[i],
            "solimp": source.geom_solimp[i],
            "margin": float(source.geom_margin[i]),
            **overrides,
        }
        if source.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = source.geom_dataid[i]
            name = source.mesh(mesh).name
            if name not in meshes:
                meshes.add(name)
                start, count = source.mesh_vertadr[mesh], source.mesh_vertnum[mesh]
                faces = source.mesh_faceadr[mesh], source.mesh_facenum[mesh]
                added = spec.add_mesh(name=name)
                added.uservert = source.mesh_vert[start : start + count].ravel()
                added.userface = source.mesh_face[
                    faces[0] : faces[0] + faces[1]
                ].ravel()
            fields["meshname"] = name
        spec.worldbody.add_geom(name=source.geom(i).name or f"{prefix}_{i}", **fields)


def workspace(spec):
    source = mujoco.MjModel.from_xml_path(
        str(ROOT / "manipulation/aloha/workspace.xml")
    )
    # mjlab 1.6 resolves joint positions by leaf name but keyframe controls
    # by full XML target name. Restore the bundled controls for slash-named arms.
    spec.keys[0].ctrl = source.key_ctrl[0].copy()
    copy_world_geoms(
        spec,
        source,
        prefix="workspace",
        priority=2,
        solref=(0.004, 1),
        solimp=(0.99, 0.99, 0.001, 0.5, 2),
    )
    for side, x in [("left", -0.14), ("right", 0.14)]:
        spec.worldbody.add_site(
            name=side + "_target",
            pos=(x, 0.23, 0.002),
            size=(0.075, 0.002, 0),
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=(0.3, 0.8, 0.55, 0.55),
        )
    spec.worldbody.add_light(pos=(0, 0, 3), dir=(0, 0, -1))


def scene_config(num_envs=1):
    entities = {
        "robot": EntityCfg(
            spec_fn=robot_spec,
            init_state=EntityCfg.InitialStateCfg(
                joint_pos={
                    "waist": 0.0,
                    "shoulder": -0.96,
                    "elbow": 1.16,
                    "forearm_roll": 0.0,
                    "wrist_angle": -0.3,
                    "wrist_rotate": 0.0,
                    "left_finger": 0.0084,
                    "right_finger": 0.0084,
                }
            ),
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    XmlActuatorCfg(
                        target_names_expr=(
                            "waist|shoulder|elbow|forearm_roll|wrist_angle|wrist_rotate|left_finger",
                        )
                    ),
                )
            ),
        ),
    }
    for i, y in enumerate((-0.12, 0.12)):
        entities[f"object_{i}"] = EntityCfg(
            spec_fn=partial(object_spec, i),
            init_state=EntityCfg.InitialStateCfg(pos=(0, y, 0.035)),
        )
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=0,
        entities=entities,
        sensors=tuple(
            ContactSensorCfg(
                name=f"object_{i}_contact",
                primary=ContactMatch(
                    mode="body", pattern=f"object_{i}", entity=f"object_{i}"
                ),
                fields=("found", "force", "dist", "pos", "normal", "tangent"),
                reduce="maxforce",
                global_frame=True,
            )
            for i in range(2)
        ),
        spec_fn=workspace,
        extent=2.0,
    )


def export_scene(env, directory):
    """Export mesh bytes and actual runtime solver options for faithful replay."""
    from mjlab.utils.spec import export_spec

    spec = env.scene.spec.copy()
    for name in dir(spec.option):
        if name.startswith("_"):
            continue
        value = getattr(env.sim.mj_model.opt, name, None)
        if value is not None and not callable(value):
            setattr(spec.option, name, value)
    export_spec(spec, directory)
