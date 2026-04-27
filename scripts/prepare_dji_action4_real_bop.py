#!/usr/bin/env python
"""Prepare the realistic DJI Action4 GLB as a BOP target model.

This is a convenience entrypoint for the wrist-occlusion dataset branch.  It
uses the same conversion code as ``prepare_dji_action4_bop.py`` but points to
``dji-action4-real`` by default, leaving the existing ``dji-action4`` dataset
untouched.
"""

from __future__ import annotations

import argparse
import sys

from prepare_dji_action4_bop import (
    _center_and_scale,
    _clear_scene,
    _export_ascii_ply,
    _import_glb,
    _resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dji-action4-real-with-hand/DJI Action4 3d model GLB into "
            "dji-action4-real/models/obj_000001.ply."
        )
    )
    parser.add_argument(
        "--input-glb",
        default="dji-action4-real-with-hand/DJI Action4 3d.glb",
        help="Path to the realistic DJI Action4 GLB, relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--output-ply",
        default="dji-action4-real/models/obj_000001.ply",
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
    obj = _import_and_export(mesh_objects, args.unit_scale, output_ply, args.overwrite)

    scale, min_mm, max_mm = obj
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


def _import_and_export(mesh_objects, unit_scale, output_ply, overwrite):
    from prepare_dji_action4_bop import _join_meshes

    obj = _join_meshes(mesh_objects)
    scale, min_mm, max_mm = _center_and_scale(obj, unit_scale)
    _export_ascii_ply(obj, output_ply, overwrite)
    return scale, min_mm, max_mm


if __name__ == "__main__":
    sys.exit(main())
