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


def _import_centered_wrist_template(
    wrist_glb: Path,
    wrist_unit_scale: str,
    wrist_size_scale: float,
    bpy: Any,
    Vector: Any,
) -> Any:
    if not wrist_glb.is_file():
        raise FileNotFoundError(f"Missing wrist GLB: {wrist_glb}")

    before_names = {obj.name for obj in bpy.context.scene.objects}
    bpy.ops.import_scene.gltf(filepath=os.fspath(wrist_glb))
    imported_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.name not in before_names and obj.type == "MESH"
    ]
    if not imported_meshes:
        raise RuntimeError(f"No mesh objects were imported from {wrist_glb}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_meshes[0]
    if len(imported_meshes) > 1:
        bpy.ops.object.join()

    template = bpy.context.view_layer.objects.active
    template.name = "wrist_occluder_template"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    min_corner, max_corner = _world_bbox(template, Vector)
    extent = max(max_corner - min_corner)
    if wrist_unit_scale == "auto":
        unit_scale = 0.001 if extent > 2.0 else 1.0
    else:
        unit_scale = float(wrist_unit_scale)
    scale = unit_scale * wrist_size_scale
    center = (min_corner + max_corner) * 0.5

    for vertex in template.data.vertices:
        world_coord = template.matrix_world @ vertex.co
        vertex.co = (world_coord - center) * scale
    template.matrix_world.identity()
    template.location = (0.0, 0.0, 0.0)
    template.rotation_euler = (0.0, 0.0, 0.0)
    template.scale = (1.0, 1.0, 1.0)
    template.data.update()
    template.hide_viewport = True
    template.hide_render = True
    return template


def _clone_wrist(template: Any, index: int, bpy: Any) -> Any:
    wrist = template.copy()
    wrist.data = template.data.copy()
    wrist.name = f"wrist_occluder_{index:02d}"
    bpy.context.collection.objects.link(wrist)
    wrist.hide_viewport = False
    wrist.hide_render = False
    return wrist


def _profile_offsets(profile: str, np: Any) -> tuple[float, float, float]:
    if profile == "light":
        return (
            float(np.random.uniform(-0.006, 0.006)),
            float(np.random.uniform(0.030, 0.050)),
            float(np.random.uniform(-0.020, -0.010)),
        )
    if profile == "heavy":
        return (
            float(np.random.uniform(-0.012, 0.012)),
            float(np.random.uniform(0.000, 0.025)),
            float(np.random.uniform(-0.010, 0.010)),
        )
    return (
        float(np.random.uniform(-0.010, 0.010)),
        float(np.random.uniform(0.012, 0.038)),
        float(np.random.uniform(-0.016, 0.004)),
    )


def _set_pair_poses(
    target_bop_objs: list[Any],
    wrist_objs: list[Any],
    occlusion_profile: str,
    np: Any,
    Matrix: Any,
    Euler: Any,
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
                np.random.uniform(0.060, 0.115),
            ],
            dtype=float,
        )
        yaw = math.radians(side * np.random.uniform(8.0, 22.0) + np.random.uniform(-7.0, 7.0))
        pitch = math.radians(np.random.uniform(-10.0, 10.0))
        roll = math.radians(side * np.random.uniform(5.0, 18.0))

        target.set_location(location)
        target.set_rotation_euler([pitch, roll, yaw])
        bpy.context.view_layer.update()
        target_blender_obj = _get_blender_object(target)

        offset = _profile_offsets(occlusion_profile, np)
        wrist_local = Matrix.Translation(offset) @ Euler(
            (
                math.radians(np.random.uniform(-8.0, 8.0)),
                math.radians(side * np.random.uniform(8.0, 18.0)),
                math.radians(np.random.uniform(-10.0, 10.0)),
            ),
            "XYZ",
        ).to_matrix().to_4x4()
        wrist.matrix_world = target_blender_obj.matrix_world @ wrist_local


def _sample_head_camera_sequence(
    frame_count: int,
    np: Any,
    bproc: Any,
) -> list[Any]:
    showcase = random.random() < 0.20
    if showcase:
        anchor_location = np.array(
            [
                np.random.uniform(-0.08, 0.08),
                np.random.uniform(-0.52, -0.34),
                np.random.uniform(0.38, 0.60),
            ],
            dtype=float,
        )
        anchor_poi = np.array(
            [
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.12, -0.04),
                np.random.uniform(0.070, 0.135),
            ],
            dtype=float,
        )
    else:
        anchor_location = np.array(
            [
                np.random.uniform(-0.12, 0.12),
                np.random.uniform(-0.74, -0.46),
                np.random.uniform(0.55, 0.85),
            ],
            dtype=float,
        )
        anchor_poi = np.array(
            [
                np.random.uniform(-0.06, 0.06),
                np.random.uniform(-0.12, 0.04),
                np.random.uniform(0.040, 0.110),
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


def render_wrist_scene(
    args: argparse.Namespace,
    material: Any,
    output_dataset_path: Path,
) -> None:
    os.environ["EGL_DEVICE_ID"] = str(args.gpu_id)

    import blenderproc as bproc
    import bpy
    import numpy as np
    from mathutils import Euler, Matrix, Vector

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
    for plane in room_planes:
        plane.replace_materials(cc_materials[0])

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
        bpy,
        Vector,
    )
    wrist_objs = [_clone_wrist(wrist_template, index, bpy) for index in range(args.object_count)]

    for obj in target_bop_objs:
        obj.set_shading_mode("auto")
        material_slot = obj.get_materials()[0]
        material_slot.set_principled_shader_value("Roughness", np.random.uniform(0.35, 0.85))
        material_slot.set_principled_shader_value("Specular", np.random.uniform(0.10, 0.45))
        obj.hide(False)

    _set_pair_poses(
        target_bop_objs,
        wrist_objs,
        args.occlusion_profile,
        np,
        Matrix,
        Euler,
        bpy,
    )

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.set_render_devices(
        desired_gpu_device_type="CUDA",
        desired_gpu_ids=[args.gpu_id],
    )

    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(target_bop_objs)
    cam_poses = 0
    for cam2world_matrix in _sample_head_camera_sequence(args.views_per_scene, np, bproc):
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.25}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    attempts = 0
    while cam_poses < args.views_per_scene and attempts < args.views_per_scene * 10:
        attempts += 1
        fallback_matrix = _sample_head_camera_sequence(1, np, bproc)[0]
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
