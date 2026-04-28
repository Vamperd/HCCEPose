#!/usr/bin/env python3
"""Render DJI Action4 wrist-occlusion scenes into a BOP train_pbr dataset.

This renderer is a sibling of ``render_dji_action4_material_scene.py``.  It keeps
the BOP target set limited to DJI Action4 (obj_id=1), while importing the wrist
GLB as render-only geometry so the wrist can occlude the camera without becoming
a training class or a corner-label target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from render_dji_action4_material_scene import (
    append_manifest_record,
    ensure_output_dataset_scaffold,
    enumerate_materials,
    expected_slot,
    get_done_indices,
    get_written_frame_count,
    latest_record_by_material,
    load_manifest_records,
    validate_material_index,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Render one material-indexed DJI Action4 wrist-occlusion scene. "
            "The wrist GLB is render-only and is never written as a BOP target."
        )
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index.")
    parser.add_argument(
        "--textures-path",
        type=Path,
        default=repo_root / "cc0textures-512",
        help="Path to cc0textures or cc0textures-512 used only for room/background materials.",
    )
    parser.add_argument(
        "--source-dataset-path",
        type=Path,
        default=repo_root / "dji-action4-real",
        help="Source BOP dataset path containing the real DJI Action4 models/ directory.",
    )
    parser.add_argument(
        "--output-dataset-path",
        type=Path,
        default=repo_root / "dji-action4-real-wrist-occlusion",
        help="Target BOP dataset path used to store wrist-occlusion train_pbr output.",
    )
    parser.add_argument(
        "--wrist-glb",
        type=Path,
        default=repo_root
        / "dji-action4-real-with-hand"
        / "wrist3d.glb",
        help="Render-only wrist GLB path.",
    )
    parser.add_argument(
        "--jacket-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the render-only jacket/background GLB.",
    )
    parser.add_argument(
        "--jacket-glb",
        type=Path,
        default=repo_root
        / "dji-action4-real-with-hand"
        / "jack3d.glb",
        help="Render-only jacket GLB path.",
    )
    parser.add_argument(
        "--material-index",
        type=int,
        default=None,
        help="Zero-based material index from the stable sorted material list.",
    )
    parser.add_argument("--object-id", type=int, default=1, help="DJI Action4 BOP object id.")
    parser.add_argument(
        "--object-count",
        type=int,
        default=2,
        help="Number of DJI Action4 target instances. Default matches left/right wrists.",
    )
    parser.add_argument(
        "--views-per-scene",
        "--views-per-material",
        dest="views_per_scene",
        type=int,
        default=20,
        help="Number of valid camera views rendered for this scene/material slot.",
    )
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=1000,
        help="BOP frames per train_pbr chunk.",
    )
    parser.add_argument(
        "--occlusion-profile",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="How strongly each wrist is placed over the DJI Action4.",
    )
    parser.add_argument(
        "--wrist-unit-scale",
        choices=["auto", "1", "0.001", "1000"],
        default="auto",
        help=(
            "Scale applied to the wrist GLB after import. Auto treats models with "
            "extent > 2 as millimeter models and scales them by 0.001."
        ),
    )
    parser.add_argument(
        "--wrist-size-scale",
        type=float,
        default=1.0,
        help="Additional multiplier for the wrist occluder after unit normalization.",
    )
    parser.add_argument(
        "--jacket-size-scale",
        type=float,
        default=1.0,
        help="Additional multiplier for the jacket/background model after unit normalization.",
    )
    parser.add_argument(
        "--jacket-top-z",
        type=float,
        default=0.02,
        help="World z height in meters where the jacket top surface is placed.",
    )
    parser.add_argument(
        "--jacket-front-up-axis",
        choices=["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"],
        default="neg_y",
        help="Local jacket axis treated as the front face normal and rotated to world +Z.",
    )
    parser.add_argument(
        "--jacket-use-scene-material",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the current background material package to the jacket surface.",
    )
    parser.add_argument(
        "--room-floor-gap",
        type=float,
        default=0.03,
        help="Vertical gap in meters between jacket top and the textured room floor.",
    )
    parser.add_argument(
        "--room-uv-tile-size",
        type=float,
        default=0.50,
        help="World-space meters per UV tile for textured room planes.",
    )
    parser.add_argument(
        "--wrist-decimate-ratio",
        type=float,
        default=0.25,
        help=(
            "Optional wrist mesh decimation ratio before duplicating wrist occluders. "
            "Use 1.0 to disable. This does not affect BOP labels because wrist is render-only."
        ),
    )
    parser.add_argument(
        "--jacket-decimate-ratio",
        type=float,
        default=0.10,
        help="Optional jacket mesh decimation ratio. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--target-side-up-prob",
        type=float,
        default=0.8,
        help="Probability that each DJI target is placed with its local left/right side facing upward.",
    )
    parser.add_argument(
        "--target-side-up-axis",
        choices=["x", "y"],
        default="y",
        help="Local target axis used for side-up poses. Use y by default; x is kept for validation.",
    )
    parser.add_argument(
        "--render-samples",
        type=int,
        default=32,
        help="Cycles samples per pixel. Lower values reduce GPU memory and render time.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["orbit", "head"],
        default="orbit",
        help="Camera sampling mode. orbit keeps pitch fixed while circling the subject.",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=0.55,
        help="Legacy orbit distance in meters. Used when --orbit-distance is omitted.",
    )
    parser.add_argument(
        "--orbit-distance",
        type=float,
        default=None,
        help="Camera-to-focus distance in meters for orbit mode. Prevents high pitch from moving too far away.",
    )
    parser.add_argument(
        "--orbit-pitch-deg",
        type=float,
        default=None,
        help=(
            "Optional fixed downward pitch angle in degrees. If omitted, orbit mode "
            "uses --orbit-pitch-sample-mode and the configured pitch distributions."
        ),
    )
    parser.add_argument(
        "--orbit-pitch-min-deg",
        type=float,
        default=20.0,
        help="Legacy minimum random orbit pitch angle. Used only by --orbit-pitch-sample-mode scene_uniform.",
    )
    parser.add_argument(
        "--orbit-pitch-max-deg",
        type=float,
        default=60.0,
        help="Legacy maximum random orbit pitch angle. Used only by --orbit-pitch-sample-mode scene_uniform.",
    )
    parser.add_argument(
        "--orbit-pitch-sample-mode",
        choices=["per_frame", "scene_uniform"],
        default="per_frame",
        help="Whether orbit pitch is sampled independently per frame or once per scene.",
    )
    parser.add_argument(
        "--orbit-high-pitch-prob",
        type=float,
        default=0.7,
        help="Probability of sampling orbit pitch from the high/downward pitch range.",
    )
    parser.add_argument(
        "--orbit-high-pitch-min-deg",
        type=float,
        default=50.0,
        help="Minimum high/downward orbit pitch in degrees.",
    )
    parser.add_argument(
        "--orbit-high-pitch-max-deg",
        type=float,
        default=89.0,
        help="Maximum high/downward orbit pitch in degrees.",
    )
    parser.add_argument(
        "--orbit-low-pitch-min-deg",
        type=float,
        default=0.0,
        help="Minimum low orbit pitch in degrees.",
    )
    parser.add_argument(
        "--orbit-low-pitch-max-deg",
        type=float,
        default=50.0,
        help="Maximum low orbit pitch in degrees.",
    )
    parser.add_argument(
        "--orbit-arc-deg",
        type=float,
        default=360.0,
        help="Azimuth arc covered by one scene. 360 means a full circle without duplicating the endpoint.",
    )
    parser.add_argument(
        "--orbit-start-deg",
        type=float,
        default=None,
        help="Optional orbit start azimuth in degrees. If omitted, a random start angle is used.",
    )
    parser.add_argument(
        "--orbit-roll-deg",
        type=float,
        default=0.0,
        help="Fixed in-plane camera roll in degrees for orbit mode.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip material indices already marked as done in the manifest.",
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="Print the stable material index mapping and exit.",
    )
    return parser.parse_args()


def _get_blender_object(obj: Any) -> Any:
    if hasattr(obj, "type") and hasattr(obj, "matrix_world") and hasattr(obj, "data"):
        return obj
    if hasattr(obj, "blender_obj"):
        return obj.blender_obj
    if hasattr(obj, "get_blender_obj"):
        return obj.get_blender_obj()
    raise AttributeError(f"Cannot access the underlying Blender object for {obj!r}")


def _world_bbox(blender_obj: Any, Vector: Any) -> tuple[Any, Any]:
    corners = [blender_obj.matrix_world @ Vector(corner) for corner in blender_obj.bound_box]
    min_corner = Vector(
        (
            min(corner.x for corner in corners),
            min(corner.y for corner in corners),
            min(corner.z for corner in corners),
        )
    )
    max_corner = Vector(
        (
            max(corner.x for corner in corners),
            max(corner.y for corner in corners),
            max(corner.z for corner in corners),
        )
    )
    return min_corner, max_corner


def _bbox_size_vector(obj: Any, Vector: Any, np: Any) -> Any:
    min_corner, max_corner = _world_bbox(_get_blender_object(obj), Vector)
    return np.asarray(
        [
            max_corner.x - min_corner.x,
            max_corner.y - min_corner.y,
            max_corner.z - min_corner.z,
        ],
        dtype=float,
    )


def _apply_planar_uvs(obj: Any, Vector: Any, uv_tile_size: float) -> None:
    if uv_tile_size <= 0.0:
        raise ValueError(f"--room-uv-tile-size must be positive, got {uv_tile_size}")

    blender_obj = _get_blender_object(obj)
    mesh = blender_obj.data
    if mesh is None or len(mesh.vertices) == 0:
        return

    world_coords = [blender_obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    spans = [
        max(coord[axis] for coord in world_coords) - min(coord[axis] for coord in world_coords)
        for axis in range(3)
    ]
    uv_axes = sorted(range(3), key=lambda axis: spans[axis], reverse=True)[:2]
    uv_layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = uv_layer

    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            world_coord = blender_obj.matrix_world @ mesh.vertices[vertex_index].co
            uv_layer.data[loop_index].uv = (
                world_coord[uv_axes[0]] / uv_tile_size,
                world_coord[uv_axes[1]] / uv_tile_size,
            )
    mesh.update()


def _configure_room_materials(room_planes: list[Any], material: Any, Vector: Any, uv_tile_size: float) -> None:
    for plane in room_planes:
        _apply_planar_uvs(plane, Vector, uv_tile_size)
        plane.replace_materials(material)


def _import_centered_wrist_template(
    wrist_glb: Path,
    wrist_unit_scale: str,
    wrist_size_scale: float,
    bproc: Any,
    bpy: Any,
    Vector: Any,
) -> Any:
    return _import_centered_render_template(
        wrist_glb,
        wrist_unit_scale,
        wrist_size_scale,
        "wrist_occluder_template",
        bproc,
        bpy,
        Vector,
    )


def _import_centered_render_template(
    glb_path: Path,
    unit_scale_arg: str,
    size_scale: float,
    object_name: str,
    bproc: Any,
    bpy: Any,
    Vector: Any,
) -> Any:
    if not glb_path.is_file():
        raise FileNotFoundError(f"Missing render-only GLB: {glb_path}")

    imported_meshes = bproc.loader.load_obj(os.fspath(glb_path))
    if not imported_meshes:
        raise RuntimeError(f"No mesh objects were imported from {glb_path}")
    template = imported_meshes[0]
    template_blender_obj = _get_blender_object(template)
    if len(imported_meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for mesh_obj in imported_meshes:
            blender_obj = _get_blender_object(mesh_obj)
            blender_obj.select_set(True)
        bpy.context.view_layer.objects.active = template_blender_obj
        bpy.ops.object.join()
        template_blender_obj = bpy.context.view_layer.objects.active
        template = bproc.object.convert_to_meshes([template_blender_obj])[0]
        print(
            f"[INFO] Joined {len(imported_meshes)} meshes from {glb_path.name} "
            f"into render-only object {object_name}."
        )
    template_blender_obj.name = object_name
    bpy.ops.object.select_all(action="DESELECT")
    template_blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = template_blender_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    min_corner, max_corner = _world_bbox(template_blender_obj, Vector)
    extent = max(max_corner - min_corner)
    if unit_scale_arg == "auto":
        unit_scale = 0.001 if extent > 2.0 else 1.0
    else:
        unit_scale = float(unit_scale_arg)
    scale = unit_scale * size_scale
    center = (min_corner + max_corner) * 0.5

    for vertex in template_blender_obj.data.vertices:
        world_coord = template_blender_obj.matrix_world @ vertex.co
        vertex.co = (world_coord - center) * scale
    template_blender_obj.matrix_world.identity()
    template_blender_obj.location = (0.0, 0.0, 0.0)
    template_blender_obj.rotation_euler = (0.0, 0.0, 0.0)
    template_blender_obj.scale = (1.0, 1.0, 1.0)
    template_blender_obj.data.update()
    template.hide(True)
    template_blender_obj.hide_viewport = True
    template_blender_obj.hide_render = True
    template_blender_obj.hide_set(True)
    return template


def _clone_wrist(template: Any, index: int) -> Any:
    wrist = template.duplicate()
    wrist_blender_obj = _get_blender_object(wrist)
    wrist_blender_obj.name = f"wrist_occluder_{index:02d}"
    wrist.hide(False)
    wrist_blender_obj.hide_viewport = False
    wrist_blender_obj.hide_render = False
    wrist_blender_obj.hide_set(False)
    return wrist


def _clone_wrist_shared_mesh(template: Any, index: int, bproc: Any, bpy: Any) -> Any:
    template_blender_obj = _get_blender_object(template)
    wrist_blender_obj = template_blender_obj.copy()
    wrist_blender_obj.data = template_blender_obj.data
    wrist_blender_obj.animation_data_clear()
    wrist_blender_obj.name = f"wrist_occluder_{index:02d}"
    bpy.context.collection.objects.link(wrist_blender_obj)
    wrist = bproc.object.convert_to_meshes([wrist_blender_obj])[0]
    wrist.hide(False)
    wrist_blender_obj.hide_viewport = False
    wrist_blender_obj.hide_render = False
    wrist_blender_obj.hide_set(False)
    return wrist


def _decimate_wrist_template(wrist_template: Any, ratio: float, bpy: Any) -> None:
    _decimate_render_template(wrist_template, ratio, bpy, "Wrist")


def _decimate_render_template(render_template: Any, ratio: float, bpy: Any, label: str) -> None:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"{label} decimate ratio must be in (0, 1], got {ratio}")
    if ratio >= 1.0:
        return

    blender_obj = _get_blender_object(render_template)
    render_template.hide(False)
    blender_obj.hide_viewport = False
    blender_obj.hide_set(False)
    before_vertices = len(blender_obj.data.vertices)
    before_faces = len(blender_obj.data.polygons)
    bpy.ops.object.select_all(action="DESELECT")
    blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = blender_obj
    modifier = blender_obj.modifiers.new(f"{label.lower()}_decimate", "DECIMATE")
    modifier.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    blender_obj.data.update()
    after_vertices = len(blender_obj.data.vertices)
    after_faces = len(blender_obj.data.polygons)
    print(
        f"[INFO] {label} decimated: "
        f"ratio={ratio:.3f} vertices={before_vertices}->{after_vertices} "
        f"faces={before_faces}->{after_faces}"
    )
    render_template.hide(True)
    blender_obj.hide_viewport = True
    blender_obj.hide_render = True
    blender_obj.hide_set(True)


def _axis_vector(axis_name: str, Vector: Any) -> Any:
    values = {
        "pos_x": (1.0, 0.0, 0.0),
        "neg_x": (-1.0, 0.0, 0.0),
        "pos_y": (0.0, 1.0, 0.0),
        "neg_y": (0.0, -1.0, 0.0),
        "pos_z": (0.0, 0.0, 1.0),
        "neg_z": (0.0, 0.0, -1.0),
    }
    if axis_name not in values:
        raise ValueError(f"Unsupported axis name: {axis_name}")
    return Vector(values[axis_name])


def _place_jacket_background(
    jacket_obj: Any,
    top_z: float,
    front_up_axis: str,
    Vector: Any,
    bpy: Any,
) -> tuple[float, float]:
    jacket_blender_obj = _get_blender_object(jacket_obj)
    jacket_obj.hide(False)
    jacket_blender_obj.hide_viewport = False
    jacket_blender_obj.hide_render = False
    jacket_blender_obj.hide_set(False)
    jacket_blender_obj.location = (0.0, 0.0, 0.0)
    jacket_blender_obj.rotation_euler = _axis_vector(front_up_axis, Vector).rotation_difference(
        Vector((0.0, 0.0, 1.0))
    ).to_euler()
    bpy.context.view_layer.update()

    _, max_corner = _world_bbox(jacket_blender_obj, Vector)
    jacket_blender_obj.location.z += top_z - max_corner.z
    bpy.context.view_layer.update()

    min_corner, max_corner = _world_bbox(jacket_blender_obj, Vector)
    return float(max_corner.z), float(min_corner.z)


def _normalize_wrist_template_scale(
    wrist_template: Any,
    reference_target: Any,
    np: Any,
    Vector: Any,
) -> None:
    target_blender_obj = _get_blender_object(reference_target)
    wrist_size = _bbox_size_vector(wrist_template, Vector, np)
    target_size = _bbox_size_vector(target_blender_obj, Vector, np)
    wrist_extent = float(np.max(wrist_size))
    target_extent = float(np.max(target_size))
    if wrist_extent <= 1e-8 or target_extent <= 1e-8:
        return

    current_ratio = wrist_extent / target_extent
    desired_ratio = 1.65
    min_ratio = 0.65
    max_ratio = 3.20
    if min_ratio <= current_ratio <= max_ratio:
        print(
            "[INFO] Wrist scale kept as-is: "
            f"wrist_extent={wrist_extent:.4f}m target_extent={target_extent:.4f}m "
            f"ratio={current_ratio:.3f}"
        )
        return

    scale_factor = desired_ratio / current_ratio
    wrist_blender_obj = _get_blender_object(wrist_template)
    for vertex in wrist_blender_obj.data.vertices:
        vertex.co *= scale_factor
    wrist_blender_obj.data.update()
    new_wrist_extent = float(np.max(_bbox_size_vector(wrist_template, Vector, np)))
    print(
        "[INFO] Wrist scale normalized: "
        f"old_extent={wrist_extent:.4f}m new_extent={new_wrist_extent:.4f}m "
        f"target_extent={target_extent:.4f}m scale_factor={scale_factor:.3f}"
    )


def _profile_offsets(profile: str, np: Any, side: float) -> tuple[float, float, float]:
    if profile == "light":
        return (
            float(side * np.random.uniform(0.040, 0.060)),
            float(np.random.uniform(-0.050, -0.028)),
            float(np.random.uniform(-0.010, 0.008)),
        )
    if profile == "heavy":
        return (
            float(side * np.random.uniform(0.015, 0.035)),
            float(np.random.uniform(-0.018, 0.010)),
            float(np.random.uniform(-0.012, 0.014)),
        )
    return (
        float(side * np.random.uniform(0.025, 0.048)),
        float(np.random.uniform(-0.035, -0.008)),
        float(np.random.uniform(-0.012, 0.010)),
    )


def _set_pair_poses(
    target_bop_objs: list[Any],
    wrist_objs: list[Any],
    occlusion_profile: str,
    jacket_top_z: float,
    target_side_up_prob: float,
    target_side_up_axis: str,
    np: Any,
    Matrix: Any,
    Euler: Any,
    Vector: Any,
    bpy: Any,
) -> None:
    count = len(target_bop_objs)
    spacing = 0.26 if count > 1 else 0.0
    start_x = -0.5 * spacing * (count - 1)

    for index, (target, wrist) in enumerate(zip(target_bop_objs, wrist_objs)):
        side = -1.0 if index % 2 == 0 else 1.0
        location = np.array(
            [
                start_x + index * spacing + np.random.uniform(-0.025, 0.025),
                np.random.uniform(-0.18, -0.08),
                jacket_top_z + np.random.uniform(0.060, 0.115),
            ],
            dtype=float,
        )
        yaw = math.radians(side * np.random.uniform(8.0, 22.0) + np.random.uniform(-7.0, 7.0))
        pitch = math.radians(np.random.uniform(-10.0, 10.0))
        roll = math.radians(side * np.random.uniform(5.0, 18.0))
        wrist_anchor_rotation = Euler((pitch, roll, yaw), "XYZ").to_matrix().to_4x4()
        target_side_up_prob = min(max(float(target_side_up_prob), 0.0), 1.0)
        if np.random.random() < target_side_up_prob:
            side_up_sign = -1.0 if np.random.random() < 0.5 else 1.0
            orientation_category = "left_side_up" if side_up_sign < 0.0 else "right_side_up"
            side_tilt = math.radians(side_up_sign * 90.0 + np.random.uniform(-8.0, 8.0))
            side_axis = "X" if target_side_up_axis == "y" else "Y"
            target_rotation_matrix = (
                Matrix.Rotation(yaw, 4, "Z")
                @ Matrix.Rotation(side_tilt, 4, side_axis)
                @ Euler(
                    (
                        math.radians(np.random.uniform(-7.0, 7.0)),
                        math.radians(np.random.uniform(-5.0, 5.0)),
                        math.radians(np.random.uniform(-6.0, 6.0)),
                    ),
                    "XYZ",
                ).to_matrix().to_4x4()
            )
            rotation_euler = target_rotation_matrix.to_euler()
        else:
            orientation_category = "top_up"
            rotation_euler = Euler((pitch, roll, yaw), "XYZ")

        target.set_location(location)
        target.set_rotation_euler(rotation_euler)
        print(
            f"[INFO] target[{index}] orientation={orientation_category} "
            f"axis={target_side_up_axis}"
        )
        bpy.context.view_layer.update()
        target_blender_obj = _get_blender_object(target)
        target_location = target_blender_obj.matrix_world.translation.copy()

        offset = _profile_offsets(occlusion_profile, np, side)
        wrist_local = Matrix.Translation(offset) @ Euler(
            (
                math.radians(np.random.uniform(-8.0, 8.0)),
                math.radians(side * np.random.uniform(8.0, 18.0)),
                math.radians(np.random.uniform(-10.0, 10.0)),
            ),
            "XYZ",
        ).to_matrix().to_4x4()
        wrist_blender_obj = _get_blender_object(wrist)
        wrist_anchor_no_scale = Matrix.Translation(target_location) @ wrist_anchor_rotation
        wrist_blender_obj.matrix_world = wrist_anchor_no_scale @ wrist_local
        wrist.hide(False)
        wrist_blender_obj.hide_viewport = False
        wrist_blender_obj.hide_render = False
        wrist_blender_obj.hide_set(False)
        bpy.context.view_layer.update()

        target_min, _ = _world_bbox(target_blender_obj, Vector)
        wrist_min, _ = _world_bbox(wrist_blender_obj, Vector)
        min_z = min(float(target_min.z), float(wrist_min.z))
        min_allowed_z = jacket_top_z + 0.005
        if min_z < min_allowed_z:
            lift = min_allowed_z - min_z
            target.set_location(target_blender_obj.location + Vector((0.0, 0.0, lift)))
            wrist_blender_obj.matrix_world.translation.z += lift
            bpy.context.view_layer.update()
            print(f"[INFO] target[{index}] wrist_pair_lifted_m={lift:.4f}")


def _sample_head_camera_sequence(
    frame_count: int,
    np: Any,
    bproc: Any,
    focus_center: Any,
) -> list[Any]:
    showcase = random.random() < 0.20
    if showcase:
        anchor_location = np.array(
            [
                focus_center[0] + np.random.uniform(-0.05, 0.05),
                focus_center[1] + np.random.uniform(-0.48, -0.32),
                focus_center[2] + np.random.uniform(0.28, 0.46),
            ],
            dtype=float,
        )
        anchor_poi = np.array(
            [
                focus_center[0] + np.random.uniform(-0.035, 0.035),
                focus_center[1] + np.random.uniform(-0.015, 0.020),
                focus_center[2] + np.random.uniform(-0.010, 0.035),
            ],
            dtype=float,
        )
    else:
        anchor_location = np.array(
            [
                focus_center[0] + np.random.uniform(-0.08, 0.08),
                focus_center[1] + np.random.uniform(-0.72, -0.44),
                focus_center[2] + np.random.uniform(0.42, 0.70),
            ],
            dtype=float,
        )
        anchor_poi = np.array(
            [
                focus_center[0] + np.random.uniform(-0.045, 0.045),
                focus_center[1] + np.random.uniform(-0.020, 0.030),
                focus_center[2] + np.random.uniform(-0.020, 0.030),
            ],
            dtype=float,
        )

    roll_anchor = math.radians(np.random.uniform(-5.0, 5.0))
    cam2world_mats = []
    for frame_index in range(frame_count):
        phase = 0.0 if frame_count <= 1 else frame_index / float(frame_count - 1)
        drift = np.array(
            [
                math.sin(phase * math.pi * 2.0) * 0.018,
                math.cos(phase * math.pi * 1.5) * 0.016,
                math.sin(phase * math.pi) * 0.012,
            ],
            dtype=float,
        )
        location = anchor_location + drift + np.random.normal(0.0, 0.010, size=3)
        poi = anchor_poi + np.random.normal(0.0, 0.014, size=3)
        roll = roll_anchor + math.radians(np.random.uniform(-3.0, 3.0))
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=roll)
        cam2world_mats.append(bproc.math.build_transformation_mat(location, rotation_matrix))
    return cam2world_mats


def _sample_orbit_camera_sequence(
    frame_count: int,
    np: Any,
    bproc: Any,
    focus_center: Any,
    distance: float,
    pitch_deg: float | None,
    pitch_min_deg: float,
    pitch_max_deg: float,
    pitch_sample_mode: str,
    high_pitch_prob: float,
    high_pitch_min_deg: float,
    high_pitch_max_deg: float,
    low_pitch_min_deg: float,
    low_pitch_max_deg: float,
    arc_deg: float,
    start_deg: float | None,
    roll_deg: float,
    jacket_top_z: float,
) -> list[Any]:
    if frame_count <= 0:
        return []
    if distance <= 0.0:
        raise ValueError(f"--orbit-distance must be positive, got {distance}")

    def validate_pitch_range(label: str, min_deg: float, max_deg: float) -> None:
        if min_deg < 0.0 or max_deg >= 89.5 or min_deg > max_deg:
            raise ValueError(
                f"{label} must satisfy 0 <= min <= max < 89.5, "
                f"got min={min_deg}, max={max_deg}"
            )

    validate_pitch_range("--orbit-pitch-min/max-deg", pitch_min_deg, pitch_max_deg)
    validate_pitch_range("--orbit-high-pitch-min/max-deg", high_pitch_min_deg, high_pitch_max_deg)
    validate_pitch_range("--orbit-low-pitch-min/max-deg", low_pitch_min_deg, low_pitch_max_deg)
    if not 0.0 <= high_pitch_prob <= 1.0:
        raise ValueError(
            f"--orbit-high-pitch-prob must be in [0, 1], got {high_pitch_prob}"
        )

    start = math.radians(float(np.random.uniform(0.0, 360.0) if start_deg is None else start_deg))
    arc = math.radians(arc_deg)
    roll = math.radians(roll_deg)

    fixed_pitch_deg = pitch_deg
    scene_pitch_deg = None
    if fixed_pitch_deg is not None:
        if fixed_pitch_deg < 0.0 or fixed_pitch_deg >= 89.5:
            raise ValueError(f"--orbit-pitch-deg must be in [0, 89.5), got {fixed_pitch_deg}")
    elif pitch_sample_mode == "scene_uniform":
        scene_pitch_deg = float(np.random.uniform(pitch_min_deg, pitch_max_deg))
        print(f"[INFO] orbit_pitch_sampled_deg scene={scene_pitch_deg:.2f} bucket=scene_uniform")

    def sample_pitch(frame_index: int) -> tuple[float, str]:
        if fixed_pitch_deg is not None:
            return float(fixed_pitch_deg), "fixed"
        if scene_pitch_deg is not None:
            return float(scene_pitch_deg), "scene_uniform"
        if np.random.random() < high_pitch_prob:
            return float(np.random.uniform(high_pitch_min_deg, high_pitch_max_deg)), "high"
        return float(np.random.uniform(low_pitch_min_deg, low_pitch_max_deg)), "low"

    cam2world_mats = []
    for frame_index in range(frame_count):
        if frame_count == 1:
            phase = 0.0
        elif abs(abs(arc_deg) - 360.0) < 1e-6:
            phase = frame_index / float(frame_count)
        else:
            phase = frame_index / float(frame_count - 1)

        theta = start + arc * phase
        sampled_pitch_deg, pitch_bucket = sample_pitch(frame_index)
        print(
            f"[INFO] orbit_pitch_sampled_deg frame={frame_index} "
            f"bucket={pitch_bucket} pitch={sampled_pitch_deg:.2f}"
        )
        horizontal = distance * math.cos(math.radians(sampled_pitch_deg))
        height = distance * math.sin(math.radians(sampled_pitch_deg))
        location = np.array(
            [
                focus_center[0] + horizontal * math.cos(theta),
                focus_center[1] + horizontal * math.sin(theta),
                max(focus_center[2] + height, jacket_top_z + 0.080),
            ],
            dtype=float,
        )
        camera_distance = float(np.linalg.norm(location - np.asarray(focus_center, dtype=float)))
        print(f"[INFO] orbit_camera_distance_m frame={frame_index} distance={camera_distance:.4f}")
        poi = np.asarray(focus_center, dtype=float)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=roll)
        cam2world_mats.append(bproc.math.build_transformation_mat(location, rotation_matrix))
    return cam2world_mats


def _sample_camera_sequence(
    args: argparse.Namespace,
    frame_count: int,
    np: Any,
    bproc: Any,
    focus_center: Any,
    jacket_top_z: float,
) -> list[Any]:
    if args.camera_mode == "orbit":
        orbit_distance = args.orbit_distance if args.orbit_distance is not None else args.orbit_radius
        return _sample_orbit_camera_sequence(
            frame_count,
            np,
            bproc,
            focus_center,
            orbit_distance,
            args.orbit_pitch_deg,
            args.orbit_pitch_min_deg,
            args.orbit_pitch_max_deg,
            args.orbit_pitch_sample_mode,
            args.orbit_high_pitch_prob,
            args.orbit_high_pitch_min_deg,
            args.orbit_high_pitch_max_deg,
            args.orbit_low_pitch_min_deg,
            args.orbit_low_pitch_max_deg,
            args.orbit_arc_deg,
            args.orbit_start_deg,
            args.orbit_roll_deg,
            jacket_top_z,
        )
    return _sample_head_camera_sequence(frame_count, np, bproc, focus_center)


def _scene_focus_center(target_bop_objs: list[Any], wrist_objs: list[Any], np: Any) -> Any:
    centers = []
    for obj in [*target_bop_objs, *wrist_objs]:
        blender_obj = _get_blender_object(obj)
        translation = blender_obj.matrix_world.translation
        centers.append([translation.x, translation.y, translation.z])
    if not centers:
        return np.array([0.0, -0.10, 0.08], dtype=float)
    return np.mean(np.asarray(centers, dtype=float), axis=0)


def _object_debug_info(obj: Any, Vector: Any, np: Any) -> tuple[Any, Any]:
    blender_obj = _get_blender_object(obj)
    min_corner, max_corner = _world_bbox(blender_obj, Vector)
    size = np.asarray(
        [
            max_corner.x - min_corner.x,
            max_corner.y - min_corner.y,
            max_corner.z - min_corner.z,
        ],
        dtype=float,
    )
    center = np.asarray(
        [
            (min_corner.x + max_corner.x) * 0.5,
            (min_corner.y + max_corner.y) * 0.5,
            (min_corner.z + max_corner.z) * 0.5,
        ],
        dtype=float,
    )
    return size, center


def _format_vec(values: Any) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values) + ")"


def _print_scene_debug(
    target_bop_objs: list[Any],
    wrist_objs: list[Any],
    jacket_obj: Any | None,
    jacket_top_z: float,
    Vector: Any,
    np: Any,
) -> None:
    target_extents = []
    wrist_extents = []
    for index, obj in enumerate(target_bop_objs):
        size, center = _object_debug_info(obj, Vector, np)
        target_extents.append(float(np.max(size)))
        print(
            f"[INFO] target[{index}] bbox_m size={_format_vec(size)} "
            f"center={_format_vec(center)}"
        )
    for index, obj in enumerate(wrist_objs):
        size, center = _object_debug_info(obj, Vector, np)
        wrist_extents.append(float(np.max(size)))
        print(
            f"[INFO] wrist[{index}] bbox_m size={_format_vec(size)} "
            f"center={_format_vec(center)}"
        )
    if jacket_obj is not None:
        jacket_blender_obj = _get_blender_object(jacket_obj)
        min_corner, max_corner = _world_bbox(jacket_blender_obj, Vector)
        size, center = _object_debug_info(jacket_obj, Vector, np)
        print(
            f"[INFO] jacket bbox_m size={_format_vec(size)} "
            f"center={_format_vec(center)} min={_format_vec([min_corner.x, min_corner.y, min_corner.z])} "
            f"max={_format_vec([max_corner.x, max_corner.y, max_corner.z])} "
            f"top_z={jacket_top_z:.4f} bottom_z={float(min_corner.z):.4f} "
            f"hide_viewport={jacket_blender_obj.hide_viewport} "
            f"hide_render={jacket_blender_obj.hide_render} "
            f"materials={len(jacket_blender_obj.data.materials)} "
            f"rotation={_format_vec(jacket_blender_obj.rotation_euler)}"
        )
    if target_extents and wrist_extents:
        ratio = float(np.mean(wrist_extents) / max(np.mean(target_extents), 1e-8))
        print(f"[INFO] wrist_target_extent_ratio={ratio:.3f}")


def render_wrist_scene(
    args: argparse.Namespace,
    material: Any,
    output_dataset_path: Path,
) -> None:
    os.environ["EGL_DEVICE_ID"] = str(args.gpu_id)

    import blenderproc as bproc
    import bpy
    import numpy as np

    mathutils = __import__("mathutils")
    Euler = mathutils.Euler
    Matrix = mathutils.Matrix
    Vector = mathutils.Vector

    bop_dataset_path = str(output_dataset_path)
    bop_parent_path = str(output_dataset_path.parent)
    dataset_name = output_dataset_path.name

    bproc.init()
    bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)

    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
        bproc.object.create_primitive(
            "PLANE",
            scale=[2, 2, 1],
            location=[0, -2, 2],
            rotation=[-1.570796, 0, 0],
        ),
        bproc.object.create_primitive(
            "PLANE",
            scale=[2, 2, 1],
            location=[0, 2, 2],
            rotation=[1.570796, 0, 0],
        ),
        bproc.object.create_primitive(
            "PLANE",
            scale=[2, 2, 1],
            location=[2, 0, 2],
            rotation=[0, -1.570796, 0],
        ),
        bproc.object.create_primitive(
            "PLANE",
            scale=[2, 2, 1],
            location=[-2, 0, 2],
            rotation=[0, 1.570796, 0],
        ),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape="BOX",
            mass=1.0,
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )

    if material.loader_kind == "cc0textures-512":
        cc_materials = bproc.loader.load_512_dict_ccmaterials(all_cc0_paths=[material.payload])
    else:
        cc_materials = bproc.loader.load_ccmaterials(
            str(args.textures_path),
            used_assets=[material.payload],
            use_all_materials=True,
        )
    if not cc_materials:
        raise RuntimeError(f"Failed to load material for index {material.index}: {material.name}")
    selected_room_material = cc_materials[0]

    light_plane = bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3.0, 6.0),
        emission_color=np.random.uniform(
            [0.5, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ),
    )
    light_plane.replace_materials(light_plane_material)
    light_point = bproc.types.Light()
    light_point.set_energy(200)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))
    light_point.set_location(
        [
            np.random.uniform(-0.45, 0.45),
            np.random.uniform(-0.65, -0.20),
            np.random.uniform(0.85, 1.45),
        ]
    )

    target_bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=bop_dataset_path,
        mm2m=True,
        obj_ids=[args.object_id] * args.object_count,
    )
    if len(target_bop_objs) != args.object_count:
        raise RuntimeError(
            f"Expected {args.object_count} DJI objects, got {len(target_bop_objs)}."
        )

    wrist_template = _import_centered_wrist_template(
        args.wrist_glb,
        args.wrist_unit_scale,
        args.wrist_size_scale,
        bproc,
        bpy,
        Vector,
    )
    _normalize_wrist_template_scale(wrist_template, target_bop_objs[0], np, Vector)
    _decimate_wrist_template(wrist_template, args.wrist_decimate_ratio, bpy)
    wrist_objs = [
        _clone_wrist_shared_mesh(wrist_template, index, bproc, bpy)
        for index in range(args.object_count)
    ]

    jacket_obj = None
    jacket_top_z = 0.0
    jacket_bottom_z = -0.02
    if args.jacket_enabled:
        jacket_obj = _import_centered_render_template(
            args.jacket_glb,
            "auto",
            args.jacket_size_scale,
            "jacket_background",
            bproc,
            bpy,
            Vector,
        )
        _decimate_render_template(jacket_obj, args.jacket_decimate_ratio, bpy, "Jacket")
        if args.jacket_use_scene_material:
            jacket_obj.replace_materials(cc_materials[0])
        jacket_top_z, jacket_bottom_z = _place_jacket_background(
            jacket_obj,
            args.jacket_top_z,
            args.jacket_front_up_axis,
            Vector,
            bpy,
        )
        room_floor_z = jacket_top_z - args.room_floor_gap
        room_planes[0].set_location([0.0, 0.0, room_floor_z])
        print(
            "[INFO] Jacket background enabled: "
            f"glb={args.jacket_glb} top_z={jacket_top_z:.4f}m "
            f"bottom_z={jacket_bottom_z:.4f}m ground_z={room_floor_z:.4f}m "
            f"front_up_axis={args.jacket_front_up_axis} "
            f"use_scene_material={args.jacket_use_scene_material}"
        )
    _configure_room_materials(room_planes, selected_room_material, Vector, args.room_uv_tile_size)
    print(
        "[INFO] Room material applied: "
        f"material={material.name} planes={len(room_planes)} "
        f"uv_tile_size={args.room_uv_tile_size:.3f}m "
        f"floor_z={_get_blender_object(room_planes[0]).location.z:.4f}m"
    )

    for obj in target_bop_objs:
        obj.set_shading_mode("auto")
        material_slot = obj.get_materials()[0]
        material_slot.set_principled_shader_value("Roughness", np.random.uniform(0.35, 0.85))
        material_slot.set_principled_shader_value("Specular", np.random.uniform(0.10, 0.45))
        obj.hide(False)
        _get_blender_object(obj).hide_render = False

    _set_pair_poses(
        target_bop_objs,
        wrist_objs,
        args.occlusion_profile,
        jacket_top_z,
        args.target_side_up_prob,
        args.target_side_up_axis,
        np,
        Matrix,
        Euler,
        Vector,
        bpy,
    )
    bpy.context.view_layer.update()
    _print_scene_debug(target_bop_objs, wrist_objs, jacket_obj, jacket_top_z, Vector, np)
    focus_center = _scene_focus_center(target_bop_objs, wrist_objs, np)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.render_samples)
    bproc.renderer.set_render_devices(
        desired_gpu_device_type="CUDA",
        desired_gpu_ids=[args.gpu_id],
    )

    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(target_bop_objs)
    cam_poses = 0
    orbit_distance = args.orbit_distance if args.orbit_distance is not None else args.orbit_radius
    print(
        "[INFO] camera_mode="
        f"{args.camera_mode} orbit_distance={orbit_distance:.3f}m "
        f"orbit_radius_legacy={args.orbit_radius:.3f}m "
        f"orbit_pitch={args.orbit_pitch_deg if args.orbit_pitch_deg is not None else 'random'} "
        f"orbit_pitch_sample_mode={args.orbit_pitch_sample_mode} "
        f"orbit_high_pitch=({args.orbit_high_pitch_min_deg:.2f}, "
        f"{args.orbit_high_pitch_max_deg:.2f})deg prob={args.orbit_high_pitch_prob:.2f} "
        f"orbit_low_pitch=({args.orbit_low_pitch_min_deg:.2f}, "
        f"{args.orbit_low_pitch_max_deg:.2f})deg "
        f"orbit_arc={args.orbit_arc_deg:.2f}deg"
    )
    for cam2world_matrix in _sample_camera_sequence(
        args,
        args.views_per_scene,
        np,
        bproc,
        focus_center,
        jacket_top_z,
    ):
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.25}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    attempts = 0
    while cam_poses < args.views_per_scene and attempts < args.views_per_scene * 10:
        attempts += 1
        fallback_matrix = _sample_camera_sequence(
            args,
            1,
            np,
            bproc,
            focus_center,
            jacket_top_z,
        )[0]
        if bproc.camera.perform_obstacle_in_view_check(fallback_matrix, {"min": 0.25}, bop_bvh_tree):
            bproc.camera.add_camera_pose(fallback_matrix, frame=cam_poses)
            cam_poses += 1
    if cam_poses != args.views_per_scene:
        raise RuntimeError(
            f"Could only sample {cam_poses}/{args.views_per_scene} valid camera poses."
        )

    data = bproc.renderer.render()
    bproc.writer.write_bop(
        bop_parent_path,
        target_objects=target_bop_objs,
        dataset=dataset_name,
        depth_scale=0.1,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format="JPEG",
        ignore_dist_thres=10,
        append_to_existing_output=True,
        frames_per_chunk=args.frames_per_chunk,
    )


def main() -> int:
    args = parse_args()
    args.textures_path = args.textures_path.expanduser().resolve()
    args.source_dataset_path = args.source_dataset_path.expanduser().resolve()
    args.output_dataset_path = args.output_dataset_path.expanduser().resolve()
    args.wrist_glb = args.wrist_glb.expanduser().resolve()
    args.jacket_glb = args.jacket_glb.expanduser().resolve()

    materials = enumerate_materials(args.textures_path)

    if args.list_materials:
        print(f"total_materials={len(materials)}")
        for material in materials:
            print(f"{material.index}\t{material.name}")
        return 0

    if args.material_index is None:
        raise SystemExit("--material-index is required unless --list-materials is used.")

    validate_material_index(args.material_index, materials)
    material = materials[args.material_index]

    ensure_output_dataset_scaffold(args.source_dataset_path, args.output_dataset_path)
    manifest_path = args.output_dataset_path / "wrist_render_manifest.jsonl"
    records = load_manifest_records(manifest_path)
    latest = latest_record_by_material(records)
    done_indices = get_done_indices(records)
    next_expected_index = len(done_indices)

    if args.material_index in latest and latest[args.material_index].get("status") == "done":
        if args.skip_done:
            print(
                f"[INFO] material-index {args.material_index} already done; "
                f"skipping ({material.name})."
            )
            return 0
        raise RuntimeError(
            f"material-index {args.material_index} is already marked as done in {manifest_path}."
        )

    if args.material_index != next_expected_index:
        raise RuntimeError(
            "This renderer appends frames sequentially. "
            f"The next expected material index is {next_expected_index}, "
            f"but got {args.material_index}. "
            "Please continue with the next contiguous material index or use a fresh output dataset."
        )

    expected_total_frames = args.material_index * args.views_per_scene
    written_frame_count = get_written_frame_count(args.output_dataset_path, args.frames_per_chunk)
    if written_frame_count != expected_total_frames:
        raise RuntimeError(
            "Output train_pbr frame count does not match the manifest expectation. "
            f"Expected {expected_total_frames} rendered frames before material {args.material_index}, "
            f"but found {written_frame_count}."
        )

    chunk_id, frame_start, frame_end = expected_slot(
        args.material_index,
        args.views_per_scene,
        args.frames_per_chunk,
    )
    base_record = {
        "material_index": args.material_index,
        "material_name": material.name,
        "chunk_id": f"{chunk_id:06d}",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "views_written": args.views_per_scene,
        "object_count": args.object_count,
        "occlusion_profile": args.occlusion_profile,
        "wrist_glb": str(args.wrist_glb),
        "wrist_size_scale": args.wrist_size_scale,
        "jacket_enabled": args.jacket_enabled,
        "jacket_glb": str(args.jacket_glb),
        "jacket_size_scale": args.jacket_size_scale,
        "jacket_top_z": args.jacket_top_z,
        "jacket_front_up_axis": args.jacket_front_up_axis,
        "jacket_use_scene_material": args.jacket_use_scene_material,
        "orbit_distance": args.orbit_distance if args.orbit_distance is not None else args.orbit_radius,
        "orbit_pitch_sample_mode": args.orbit_pitch_sample_mode,
        "orbit_high_pitch_prob": args.orbit_high_pitch_prob,
        "target_side_up_prob": args.target_side_up_prob,
        "target_side_up_axis": args.target_side_up_axis,
        "output_dataset_path": str(args.output_dataset_path),
    }

    append_manifest_record(
        manifest_path,
        {"timestamp": utc_now(), "status": "started", **base_record},
    )
    print(
        "[INFO] Rendering wrist scene "
        f"material-index={args.material_index} "
        f"material={material.name} "
        f"chunk={chunk_id:06d} "
        f"frames={frame_start:04d}-{frame_end:04d}"
    )
    render_wrist_scene(args, material, args.output_dataset_path)

    append_manifest_record(
        manifest_path,
        {"timestamp": utc_now(), "status": "done", **base_record},
    )
    print(
        f"[INFO] Completed wrist material-index {args.material_index}; "
        f"wrote {args.views_per_scene} frames into chunk {chunk_id:06d}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
