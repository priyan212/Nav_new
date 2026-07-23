"""Bake a large inverted, vertex-colored sphere (a sky dome) as a GLB, in
world-space coordinates already centered over the Indian Bend & Pima scene.

Habitat-sim's raw `backend.scene_id = <path>` STAGE loader always assumes
Z-up input (see EARTH/scripts/glb_yup_to_zup.py), but *objects* added via the
ObjectAttributesManager/RigidObjectManager (the same API habitat_sim_node.py
uses for the Marsyard rocks) use the standard Y-up convention with no extra
rotation -- so this mesh is authored directly in Y-up world coordinates, no
axis fix needed. World-space translation is baked into the vertices (rather
than set at runtime) to match how the pre-placed rock meshes work.

The scene has zero configured lights ("Lighting Layout Attributes 'no_lights'"
-- see EARTH/README or habitat_sim_node.py logs), so a plain glTF mesh with no
material renders pitch black (the default material is fully metallic, which
needs an environment light to show anything). trimesh has no first-class way
to write KHR_materials_unlit, so after exporting we patch the GLB's JSON
chunk directly to add an unlit material -- that's the only part of this
script that isn't just "call trimesh".

Run inside `mars_habitat`:  python make_skydome.py
Writes EARTH/data/sky_dome.glb
"""
import json
import os
import struct

import numpy as np
import trimesh

EARTH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(EARTH_DIR, "data", "sky_dome.glb")

# Surveyed world bounds (habitat_sim_node.py): center the dome over the scene,
# at ground level, with a radius far bigger than the ~150x330 m footprint so
# it reads as "background" rather than a nearby object.
CENTER = np.array([21.5, 20.0, 4.0], dtype=np.float64)
RADIUS = 450.0

# Boosted well past the visually-intended values: habitat-sim renders this
# unlit vertex-colored mesh noticeably darker than the authored RGB (empirically
# ~0.4-0.5x per channel, and unevenly across channels -- not a simple linear
# scale, so these were tuned by re-rendering and comparing, not computed).
ZENITH = np.array([90, 170, 255], dtype=np.float64)      # saturated, sunnier azure overhead
HORIZON = np.array([255, 250, 225], dtype=np.float64)    # warm hazy near the horizon, not gray

# A visible sun disc + glow, not just a bright gradient -- direction picked so
# it's roughly in frame from the default rover start pose/heading (facing -Z).
SUN_DIR = np.array([0.25, 0.6, -0.75])
SUN_DIR /= np.linalg.norm(SUN_DIR)
SUN_COLOR = np.array([255, 250, 210], dtype=np.float64)  # bright warm white-yellow
SUN_CORE_DEG = 6.0     # solid bright disc radius
SUN_GLOW_DEG = 28.0    # soft corona radius around it


def make_unlit(glb_path: str) -> None:
    """Patch the GLB's JSON chunk in place: add a KHR_materials_unlit
    material (base color white, so it doesn't tint the COLOR_0 vertex
    colors) and point the mesh primitive at it."""
    with open(glb_path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"

    off = 12
    json_chunk = bin_chunk = None
    json_start = json_len = None
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk_start = off + 8
        chunk = data[chunk_start:chunk_start + clen]
        if ctype == 0x4E4F534A:  # 'JSON'
            json_chunk, json_start, json_len = chunk, off, clen
        elif ctype == 0x004E4942:  # 'BIN\0'
            bin_chunk = chunk
        off = chunk_start + clen

    gltf = json.loads(json_chunk)
    gltf.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in gltf["extensionsUsed"]:
        gltf["extensionsUsed"].append("KHR_materials_unlit")
    gltf["materials"] = [{
        "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]},
        "extensions": {"KHR_materials_unlit": {}},
    }]
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            prim["material"] = 0

    new_json = json.dumps(gltf).encode("utf-8")
    pad = (-len(new_json)) % 4
    new_json += b" " * pad  # glTF JSON chunks are padded with spaces

    out = bytearray()
    out += struct.pack("<4sII", b"glTF", version, 0)  # length patched below
    out += struct.pack("<II", len(new_json), 0x4E4F534A)
    out += new_json
    if bin_chunk is not None:
        bin_padded = bin_chunk + b"\x00" * ((-len(bin_chunk)) % 4)
        out += struct.pack("<II", len(bin_padded), 0x004E4942)
        out += bin_padded
    struct.pack_into("<I", out, 8, len(out))

    with open(glb_path, "wb") as f:
        f.write(out)


def main() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=RADIUS)
    sphere.invert()  # flip winding/normals so the *inside* surface is front-facing

    directions = sphere.vertices / RADIUS  # unit vectors, pre-translation
    y = directions[:, 1]                    # -1 (bottom) .. +1 (top)
    t = np.clip(y / 0.6, 0.0, 1.0)          # reaches full zenith color a bit above the horizon
    colors = HORIZON[None, :] * (1 - t[:, None]) + ZENITH[None, :] * t[:, None]

    # sun disc + glow: cosine angle to SUN_DIR -> bright core, soft falloff corona
    cos_angle = directions @ SUN_DIR
    core = cos_angle > np.cos(np.radians(SUN_CORE_DEG))
    glow = np.clip(
        (cos_angle - np.cos(np.radians(SUN_GLOW_DEG)))
        / (np.cos(np.radians(SUN_CORE_DEG)) - np.cos(np.radians(SUN_GLOW_DEG))),
        0.0, 1.0,
    ) ** 2
    colors = colors * (1 - glow[:, None]) + SUN_COLOR[None, :] * glow[:, None]
    colors[core] = SUN_COLOR

    colors = colors.astype(np.uint8)
    alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
    sphere.visual = trimesh.visual.color.ColorVisuals(
        sphere, vertex_colors=np.concatenate([colors, alpha], axis=1)
    )

    sphere.vertices += CENTER
    print(f"sky dome: {len(sphere.vertices)} verts, bounds:\n{sphere.bounds}")
    sphere.export(OUT)
    make_unlit(OUT)
    print(f"wrote {OUT} (patched with KHR_materials_unlit)")


if __name__ == "__main__":
    main()
