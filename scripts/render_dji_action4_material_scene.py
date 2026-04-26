#!/usr/bin/env python3
"""Render one DJI Action4 material scene into a BOP train_pbr dataset.

This script is intentionally designed for remote Linux rendering, where each
invocation processes exactly one material and then exits. That keeps Blender,
materials, and physics state short-lived, which is helpful when the full
material library would otherwise exhaust memory.

Expected workflow:
1. Run materials in ascending contiguous order.
2. Each material renders `views_per_material` frames with exactly two
   `object_id` instances.
3. Every 50 materials fill one 1000-frame `train_pbr/000xyz` chunk.

The manifest is append-only. A finished material receives a `status=done`
record that includes the chunk and frame slot it occupies.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CAMERA = {
    "cx": 325.2611083984375,
    "cy": 242.04899588216654,
    "depth_scale": 0.1,
    "fx": 572.411363389757,
    "fy": 573.5704328585578,
    "height": 480,
    "width": 640,
}


@dataclass(frozen=True)
class MaterialEntry:
    index: int
    name: str
    loader_kind: str
    payload: Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Render one material-specific DJI Action4 scene. "
            "Use --list-materials to inspect the stable material index mapping."
        )
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU index.")
    parser.add_argument(
        "--textures-path",
        type=Path,
        default=repo_root / "cc0textures-512",
        help="Path to cc0textures or cc0textures-512.",
    )
    parser.add_argument(
        "--source-dataset-path",
        type=Path,
        default=repo_root / "dji-action4",
        help="Source BOP dataset path containing models/ and models_info.json.",
    )
    parser.add_argument(
        "--output-dataset-path",
        type=Path,
        default=repo_root / "dji-action4-twoobj-materials",
        help="Target BOP dataset path used to store train_pbr output.",
    )
    parser.add_argument(
        "--material-index",
        type=int,
        default=None,
        help="Zero-based material index from the stable sorted material list.",
    )
    parser.add_argument("--object-id", type=int, default=1, help="Object id to render.")
    parser.add_argument(
        "--object-count",
        type=int,
        default=2,
        help="Number of DJI Action4 instances per scene.",
    )
    parser.add_argument(
        "--views-per-material",
        type=int,
        default=20,
        help="Number of valid views rendered for one material scene.",
    )
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=1000,
        help="BOP frames per train_pbr chunk.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip material indices already marked as done in the manifest.",
    )
    parser.add_argument(
        "--disable-screen-texture",
        action="store_true",
        help=(
            "Do not add a randomized current-material texture to the DJI Action4 "
            "screen face bounded by corners 1, 2, 5, and 6."
        ),
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="Print the stable material index mapping and exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_manifest_record(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in manifest {manifest_path} at line {line_number}: {exc}"
                ) from exc
    return records


def latest_record_by_material(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in records:
        material_index = record.get("material_index")
        if isinstance(material_index, int):
            latest[material_index] = record
    return latest


def get_done_indices(records: list[dict[str, Any]]) -> list[int]:
    latest = latest_record_by_material(records)
    done_indices = sorted(
        index for index, record in latest.items() if record.get("status") == "done"
    )
    for expected, actual in enumerate(done_indices):
        if actual != expected:
            raise RuntimeError(
                "Manifest contains a gap in completed materials. "
                f"Expected done material {expected}, found {actual}. "
                "Please continue from a fresh output directory or repair the manifest/output first."
            )
    return done_indices


def get_written_frame_count(output_dataset_path: Path, frames_per_chunk: int) -> int:
    train_pbr_path = output_dataset_path / "train_pbr"
    if not train_pbr_path.exists():
        return 0

    chunk_dirs = sorted(path for path in train_pbr_path.iterdir() if path.is_dir())
    if not chunk_dirs:
        return 0

    last_chunk = chunk_dirs[-1]
    scene_gt_path = last_chunk / "scene_gt.json"
    if not scene_gt_path.exists():
        raise RuntimeError(f"Missing scene_gt.json in chunk: {last_chunk}")

    with scene_gt_path.open("r", encoding="utf-8") as handle:
        scene_gt = json.load(handle)

    if not scene_gt:
        return int(last_chunk.name) * frames_per_chunk

    frame_ids = sorted(int(frame_id) for frame_id in scene_gt.keys())
    return int(last_chunk.name) * frames_per_chunk + frame_ids[-1] + 1


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return
    except OSError:
        pass

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def ensure_output_dataset_scaffold(source_dataset_path: Path, output_dataset_path: Path) -> None:
    source_models_path = source_dataset_path / "models"
    source_models_info = source_models_path / "models_info.json"
    if not source_models_path.is_dir():
        raise FileNotFoundError(f"Missing source models directory: {source_models_path}")
    if not source_models_info.is_file():
        raise FileNotFoundError(f"Missing source models_info.json: {source_models_info}")

    output_dataset_path.mkdir(parents=True, exist_ok=True)
    (output_dataset_path / "train_pbr").mkdir(exist_ok=True)

    safe_link_or_copy(source_models_path, output_dataset_path / "models")

    source_camera = source_dataset_path / "camera.json"
    output_camera = output_dataset_path / "camera.json"
    if source_camera.is_file():
        safe_link_or_copy(source_camera, output_camera)
    elif not output_camera.exists():
        with output_camera.open("w", encoding="utf-8") as handle:
            json.dump(DEFAULT_CAMERA, handle, indent=2, sort_keys=True)


def enumerate_materials(textures_path: Path) -> list[MaterialEntry]:
    if not textures_path.exists():
        raise FileNotFoundError(f"Textures path does not exist: {textures_path}")

    if textures_path.name == "cc0textures-512":
        color_files = sorted(
            (path for path in textures_path.rglob("*color.jpg") if path.is_file()),
            key=lambda path: path.relative_to(textures_path).as_posix().lower(),
        )
        materials = []
        for index, color_path in enumerate(color_files):
            relative_name = color_path.relative_to(textures_path).as_posix()
            materials.append(
                MaterialEntry(
                    index=index,
                    name=relative_name,
                    loader_kind="cc0textures-512",
                    payload={"c": str(color_path)},
                )
            )
        return materials

    material_dirs = sorted(
        (path for path in textures_path.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )
    materials = []
    for path in material_dirs:
        color_probe = path / f"{path.name}_2K-JPG_Color.jpg"
        if not color_probe.is_file():
            continue
        materials.append(
            MaterialEntry(
                index=len(materials),
                name=path.name,
                loader_kind="cc0textures",
                payload=path.name,
            )
        )
    return materials


def expected_slot(material_index: int, views_per_material: int, frames_per_chunk: int) -> tuple[int, int, int]:
    materials_per_chunk = frames_per_chunk // views_per_material
    if materials_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be greater than or equal to views_per_material.")
    if frames_per_chunk % views_per_material != 0:
        raise ValueError(
            "frames_per_chunk must be divisible by views_per_material so each material occupies a fixed slot."
        )
    chunk_id = material_index // materials_per_chunk
    slot_in_chunk = material_index % materials_per_chunk
    frame_start = slot_in_chunk * views_per_material
    frame_end = frame_start + views_per_material - 1
    return chunk_id, frame_start, frame_end


def validate_material_index(material_index: int, materials: list[MaterialEntry]) -> None:
    if material_index < 0 or material_index >= len(materials):
        raise IndexError(
            f"material-index {material_index} is out of range. "
            f"Available material indices: 0..{len(materials) - 1}"
        )


def get_material_color_path(textures_path: Path, material: MaterialEntry) -> Path:
    if material.loader_kind == "cc0textures-512":
        color_path = Path(material.payload["c"])
    elif material.loader_kind == "cc0textures":
        color_path = textures_path / material.name / f"{material.name}_2K-JPG_Color.jpg"
    else:
        raise ValueError(f"Unsupported material loader kind: {material.loader_kind}")

    if not color_path.is_file():
        raise FileNotFoundError(
            f"Missing color image for material {material.index} ({material.name}): {color_path}"
        )
    return color_path


def render_material_scene(
    args: argparse.Namespace,
    material: MaterialEntry,
    output_dataset_path: Path,
) -> None:
    os.environ["EGL_DEVICE_ID"] = str(args.gpu_id)

    import blenderproc as bproc
    import bmesh
    import bpy
    import numpy as np

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

    light_plane = bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_point = bproc.types.Light()
    light_point.set_energy(200)

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
    selected_material = cc_materials[0]
    screen_texture_path = None
    if not args.disable_screen_texture:
        screen_texture_path = get_material_color_path(args.textures_path, material)

    object_ids = [args.object_id] * args.object_count
    target_bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=bop_dataset_path,
        mm2m=True,
        obj_ids=object_ids,
    )

    def sample_pose_func(obj: Any) -> None:
        min_xyz = np.random.uniform([-0.15, -0.15, 0.0], [-0.1, -0.1, 0.0])
        max_xyz = np.random.uniform([0.1, 0.1, 0.4], [0.15, 0.15, 0.6])
        obj.set_location(np.random.uniform(min_xyz, max_xyz))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    def get_blender_object(obj: Any) -> Any:
        if hasattr(obj, "blender_obj"):
            return obj.blender_obj
        if hasattr(obj, "get_blender_obj"):
            return obj.get_blender_obj()
        raise AttributeError(f"Cannot access the underlying Blender object for {obj!r}")

    def create_screen_material(color_path: Path) -> Any:
        image = bpy.data.images.load(str(color_path), check_existing=True)
        material_name = f"dji_screen_{material.index:06d}"
        screen_material = bpy.data.materials.new(material_name)
        screen_material.use_nodes = True

        nodes = screen_material.node_tree.nodes
        links = screen_material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        uv_node = nodes.new(type="ShaderNodeUVMap")
        uv_node.uv_map = "UVMap"
        texture_node = nodes.new(type="ShaderNodeTexImage")
        texture_node.image = image
        texture_node.extension = "CLIP"
        texture_node.interpolation = "Linear"
        links.new(uv_node.outputs["UV"], texture_node.inputs["Vector"])
        if bsdf is not None:
            links.new(texture_node.outputs["Color"], bsdf.inputs["Base Color"])
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.35
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.2
            elif "Specular" in bsdf.inputs:
                bsdf.inputs["Specular"].default_value = 0.2
        return screen_material

    def randomized_screen_uvs() -> list[tuple[float, float]]:
        crop_w = float(np.random.uniform(0.35, 1.0))
        crop_h = float(np.random.uniform(0.35, 1.0))
        u0 = float(np.random.uniform(0.0, 1.0 - crop_w))
        v0 = float(np.random.uniform(0.0, 1.0 - crop_h))
        u1 = u0 + crop_w
        v1 = v0 + crop_h
        uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

        if random.random() < 0.5:
            uvs = [(u1 - (u - u0), v) for u, v in uvs]
        if random.random() < 0.5:
            uvs = [(u, v1 - (v - v0)) for u, v in uvs]
        for _ in range(random.randrange(4)):
            uvs = uvs[1:] + uvs[:1]
        return uvs

    def add_randomized_screen_face(obj: Any, screen_material: Any) -> None:
        blender_obj = get_blender_object(obj)
        mesh = blender_obj.data
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError(f"Object {blender_obj.name} has no mesh vertices.")

        blender_obj.data = mesh.copy()
        mesh = blender_obj.data
        mesh.materials.append(screen_material)
        screen_material_index = len(mesh.materials) - 1

        coords = [vertex.co.copy() for vertex in mesh.vertices]
        min_y = min(coord.y for coord in coords)
        max_y = max(coord.y for coord in coords)
        min_z = min(coord.z for coord in coords)
        max_z = max(coord.z for coord in coords)
        max_x = max(coord.x for coord in coords)
        extent = max(
            max(coord.x for coord in coords) - min(coord.x for coord in coords),
            max_y - min_y,
            max_z - min_z,
        )
        epsilon = max(extent * 1e-4, 1e-7)
        x = max_x + epsilon

        bm = bmesh.new()
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
        verts = [
            bm.verts.new((x, min_y, min_z)),
            bm.verts.new((x, max_y, min_z)),
            bm.verts.new((x, max_y, max_z)),
            bm.verts.new((x, min_y, max_z)),
        ]
        bm.verts.ensure_lookup_table()
        screen_face = bm.faces.new(verts)
        screen_face.material_index = screen_material_index
        for loop, uv in zip(screen_face.loops, randomized_screen_uvs()):
            loop[uv_layer].uv = uv
        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()

        mesh.update()

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.set_render_devices(
        desired_gpu_device_type="CUDA",
        desired_gpu_ids=[args.gpu_id],
    )

    for obj in target_bop_objs:
        obj.set_shading_mode("auto")
        obj.hide(True)

    screen_material = None
    if screen_texture_path is not None:
        screen_material = create_screen_material(screen_texture_path)

    for obj in target_bop_objs:
        if screen_material is not None:
            add_randomized_screen_face(obj, screen_material)
        material_slot = obj.get_materials()[0]
        material_slot.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
        material_slot.set_principled_shader_value("Specular", np.random.uniform(0.0, 1.0))
        obj.enable_rigidbody(
            True,
            mass=1.0,
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
        obj.hide(False)

    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3.0, 6.0),
        emission_color=np.random.uniform(
            [0.5, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ),
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))
    light_point.set_location(
        bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=1.0,
            radius_max=1.5,
            elevation_min=5.0,
            elevation_max=89.0,
        )
    )
    for plane in room_planes:
        plane.replace_materials(selected_material)

    bproc.object.sample_poses(
        objects_to_sample=target_bop_objs,
        sample_pose_func=sample_pose_func,
        max_tries=1000,
    )
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=10,
        check_object_interval=1,
        substeps_per_frame=20,
        solver_iters=25,
    )

    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(target_bop_objs)
    cam_poses = 0
    while cam_poses < args.views_per_material:
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=0.3,
            radius_max=1.2,
            elevation_min=5.0,
            elevation_max=89.0,
        )
        sample_count = max(1, min(len(target_bop_objs), int(round(0.6 * len(target_bop_objs)))))
        poi_objects = random.sample(target_bop_objs, k=sample_count)
        poi = bproc.object.compute_poi(poi_objects)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location,
            inplane_rot=np.random.uniform(-3.14159, 3.14159),
        )
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

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
    manifest_path = args.output_dataset_path / "material_render_manifest.jsonl"
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

    expected_total_frames = args.material_index * args.views_per_material
    written_frame_count = get_written_frame_count(args.output_dataset_path, args.frames_per_chunk)
    if written_frame_count != expected_total_frames:
        raise RuntimeError(
            "Output train_pbr frame count does not match the manifest expectation. "
            f"Expected {expected_total_frames} rendered frames before material {args.material_index}, "
            f"but found {written_frame_count}."
        )

    chunk_id, frame_start, frame_end = expected_slot(
        args.material_index,
        args.views_per_material,
        args.frames_per_chunk,
    )

    started_record = {
        "timestamp": utc_now(),
        "status": "started",
        "material_index": args.material_index,
        "material_name": material.name,
        "chunk_id": f"{chunk_id:06d}",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "views_written": args.views_per_material,
        "output_dataset_path": str(args.output_dataset_path),
    }
    append_manifest_record(manifest_path, started_record)

    print(
        "[INFO] Rendering "
        f"material-index={args.material_index} "
        f"material={material.name} "
        f"chunk={chunk_id:06d} "
        f"frames={frame_start:04d}-{frame_end:04d}"
    )
    render_material_scene(args, material, args.output_dataset_path)

    done_record = {
        "timestamp": utc_now(),
        "status": "done",
        "material_index": args.material_index,
        "material_name": material.name,
        "chunk_id": f"{chunk_id:06d}",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "views_written": args.views_per_material,
        "output_dataset_path": str(args.output_dataset_path),
    }
    append_manifest_record(manifest_path, done_record)
    print(
        f"[INFO] Completed material-index {args.material_index}; "
        f"wrote {args.views_per_material} frames into chunk {chunk_id:06d}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
