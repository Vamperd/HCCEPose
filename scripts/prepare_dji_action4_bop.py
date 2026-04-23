#!/usr/bin/env python
# Prepare the DJI Action4 GLB as a BOP-style PLY model for HCCEPose.

import argparse
import os
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:  # pragma: no cover - depends on Blender/bpy runtime.
    raise SystemExit(
        "This script requires bpy. Run it in the HCCEPose training environment "
        "after installing bpy, or with Blender's Python."
    ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_glb(glb_path: Path) -> list:
    bpy.ops.import_scene.gltf(filepath=os.fspath(glb_path))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects were imported from {glb_path}")
    return mesh_objects


def _join_meshes(mesh_objects: list):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = "obj_000001"
    joined.data.name = "obj_000001_mesh"
    return joined


def _world_bbox(obj) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_corner = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_corner, max_corner


def _scale_from_bbox(unit_scale: str, min_corner: Vector, max_corner: Vector) -> float:
    if unit_scale != "auto":
        return float(unit_scale)
    max_extent = max(max_corner - min_corner)
    return 1000.0 if max_extent < 2.0 else 1.0


def _center_and_scale(obj, unit_scale: str) -> tuple[float, Vector, Vector]:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    min_corner, max_corner = _world_bbox(obj)
    center = (min_corner + max_corner) * 0.5
    scale = _scale_from_bbox(unit_scale, min_corner, max_corner)

    for vertex in obj.data.vertices:
        world_coord = obj.matrix_world @ vertex.co
        vertex.co = (world_coord - center) * scale

    obj.matrix_world.identity()
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    obj.data.update()

    min_mm, max_mm = _world_bbox(obj)
    return scale, min_mm, max_mm


def _export_ascii_ply(obj, output_ply: Path, overwrite: bool) -> None:
    if output_ply.exists() and not overwrite:
        raise FileExistsError(
            f"{output_ply} already exists. Pass --overwrite if you intentionally want to replace it."
        )
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    export_kwargs = {
        "filepath": os.fspath(output_ply),
        "use_selection": True,
        "use_ascii": True,
        "use_normals": True,
        "use_uv_coords": True,
        "use_colors": True,
    }
    try:
        bpy.ops.export_mesh.ply(**export_kwargs)
    except TypeError:
        export_kwargs.pop("use_selection", None)
        bpy.ops.export_mesh.ply(**export_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dji-action4/Action4_muti.glb into BOP-style models/obj_000001.ply."
    )
    parser.add_argument(
        "--input-glb",
        default="dji-action4/Action4_muti.glb",
        help="Path to the DJI Action4 GLB model, relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--output-ply",
        default="dji-action4/models/obj_000001.ply",
        help="BOP-style ASCII PLY output path, relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--unit-scale",
        default="auto",
        choices=["auto", "1", "1000"],
        help="Use auto to scale meter-sized GLB models to millimeters; use 1 or 1000 to force it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output PLY.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_glb = _resolve_path(args.input_glb)
    output_ply = _resolve_path(args.output_ply)

    if not input_glb.exists():
        raise FileNotFoundError(f"Input GLB does not exist: {input_glb}")

    _clear_scene()
    mesh_objects = _import_glb(input_glb)
    obj = _join_meshes(mesh_objects)
    scale, min_mm, max_mm = _center_and_scale(obj, args.unit_scale)
    _export_ascii_ply(obj, output_ply, args.overwrite)

    size_mm = max_mm - min_mm
    print(f"[INFO] input_glb: {input_glb}")
    print(f"[INFO] output_ply: {output_ply}")
    print(f"[INFO] applied_unit_scale: {scale:g}")
    print(
        "[INFO] bbox_mm: "
        f"min=({min_mm.x:.3f}, {min_mm.y:.3f}, {min_mm.z:.3f}), "
        f"max=({max_mm.x:.3f}, {max_mm.y:.3f}, {max_mm.z:.3f}), "
        f"size=({size_mm.x:.3f}, {size_mm.y:.3f}, {size_mm.z:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
