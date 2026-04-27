#!/usr/bin/env python
"""Prepare the realistic DJI Action4 GLB as a BOP target model.

This is a convenience entrypoint for the wrist-occlusion dataset branch.  It
uses the same conversion code as ``prepare_dji_action4_bop.py`` but points to
``dji-action4-real`` by default, leaving the existing ``dji-action4`` dataset
untouched.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

from prepare_dji_action4_bop import (
    _center_and_scale,
    _clear_scene,
    _export_ascii_ply,
    _import_glb,
    _resolve_path,
)


_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK = 0x004E4942


def _read_glb(glb_path: Path) -> tuple[dict, bytes]:
    data = glb_path.read_bytes()
    if len(data) < 12:
        raise RuntimeError(f"Invalid GLB file, too small: {glb_path}")

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise RuntimeError(f"Unsupported GLB header in {glb_path}: magic={magic!r}, version={version}")
    if length != len(data):
        raise RuntimeError(f"GLB length mismatch in {glb_path}: header={length}, actual={len(data)}")

    gltf_json = None
    bin_chunk = b""
    offset = 12
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == _GLB_JSON_CHUNK:
            gltf_json = json.loads(chunk.decode("utf-8"))
        elif chunk_type == _GLB_BIN_CHUNK:
            bin_chunk = chunk

    if gltf_json is None:
        raise RuntimeError(f"Missing JSON chunk in GLB: {glb_path}")
    return gltf_json, bin_chunk


def _basecolor_image_index(gltf_json: dict) -> int:
    materials = gltf_json.get("materials", [])
    textures = gltf_json.get("textures", [])
    images = gltf_json.get("images", [])

    for material in materials:
        pbr = material.get("pbrMetallicRoughness", {})
        texture_ref = pbr.get("baseColorTexture")
        if not texture_ref:
            continue
        texture_index = texture_ref.get("index")
        if texture_index is None or texture_index >= len(textures):
            continue
        image_index = textures[texture_index].get("source")
        if image_index is not None and image_index < len(images):
            return image_index

    for index, image in enumerate(images):
        image_name = str(image.get("name", "")).lower()
        image_uri = str(image.get("uri", "")).lower()
        if "basecolor" in image_name or "base_color" in image_name or "basecolor" in image_uri:
            return index

    if images:
        return 0
    raise RuntimeError("No images found in GLB; cannot extract a basecolor texture.")


def _image_extension(image: dict) -> str:
    mime_type = image.get("mimeType")
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"

    uri = image.get("uri")
    if uri and not uri.startswith("data:"):
        suffix = Path(uri).suffix
        if suffix:
            return suffix
    return ".bin"


def _image_bytes(glb_path: Path, gltf_json: dict, bin_chunk: bytes, image: dict) -> bytes:
    uri = image.get("uri")
    if uri:
        if uri.startswith("data:"):
            _, encoded = uri.split(",", 1)
            return base64.b64decode(encoded)
        return (glb_path.parent / uri).read_bytes()

    buffer_view_index = image.get("bufferView")
    if buffer_view_index is None:
        raise RuntimeError(f"Image has neither URI nor bufferView: {image}")

    buffer_views = gltf_json.get("bufferViews", [])
    if buffer_view_index >= len(buffer_views):
        raise RuntimeError(f"Image bufferView index out of range: {buffer_view_index}")

    buffer_view = buffer_views[buffer_view_index]
    if buffer_view.get("buffer", 0) != 0:
        raise RuntimeError(f"Only GLB buffer 0 is supported for image extraction: {buffer_view}")
    byte_offset = int(buffer_view.get("byteOffset", 0))
    byte_length = int(buffer_view["byteLength"])
    return bin_chunk[byte_offset : byte_offset + byte_length]


def _extract_basecolor_texture(input_glb: Path, output_ply: Path, overwrite: bool) -> Path:
    gltf_json, bin_chunk = _read_glb(input_glb)
    images = gltf_json.get("images", [])
    image_index = _basecolor_image_index(gltf_json)
    image = images[image_index]
    texture_path = output_ply.with_name(f"{output_ply.stem}_basecolor{_image_extension(image)}")

    if not texture_path.exists() or overwrite:
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        texture_path.write_bytes(_image_bytes(input_glb, gltf_json, bin_chunk, image))

    return texture_path


def _add_texturefile_comment(output_ply: Path, texture_path: Path) -> None:
    relative_texture = texture_path.name
    text = output_ply.read_text(encoding="latin-1")
    lines = text.splitlines()
    if not lines or lines[0] != "ply":
        raise RuntimeError(f"Unexpected PLY header in {output_ply}")

    texture_comment = f"comment TextureFile {relative_texture}"
    inserted = False
    replaced = False
    new_lines = []
    for line in lines:
        if line.startswith("comment TextureFile "):
            if not replaced:
                new_lines.append(texture_comment)
                replaced = True
            continue
        new_lines.append(line)
        if line.startswith("format ") and not replaced and not inserted:
            new_lines.append(texture_comment)
            inserted = True

    if not inserted and not replaced:
        raise RuntimeError(f"Could not find PLY format line in {output_ply}")
    output_ply.write_text("\n".join(new_lines) + "\n", encoding="latin-1")


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
    parser.add_argument(
        "--disable-texture-extraction",
        action="store_true",
        help="Do not extract the GLB basecolor texture or add a TextureFile comment to the PLY.",
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
    texture_path = None
    if not args.disable_texture_extraction:
        texture_path = _extract_basecolor_texture(input_glb, output_ply, args.overwrite)
        _add_texturefile_comment(output_ply, texture_path)

    scale, min_mm, max_mm = obj
    size_mm = max_mm - min_mm
    print(f"[INFO] input_glb: {input_glb}")
    print(f"[INFO] output_ply: {output_ply}")
    if texture_path is not None:
        print(f"[INFO] output_texture: {texture_path}")
        print(f"[INFO] ply_texture_comment: comment TextureFile {texture_path.name}")
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
