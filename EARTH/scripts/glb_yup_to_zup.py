"""Rotate a standard Y-up glTF/GLB (Sketchfab convention) into the Z-up
coordinates habitat-sim's raw scene_id loader actually expects.

Empirically (see MARS/scripts/obj2glb_trimesh.py): habitat-sim's direct
`backend.scene_id = <path>` loader always treats input vertex data as Z-up
and rotates it into its own Y-up world. A proper Y-up glTF (the normal
Sketchfab export) therefore renders standing on its edge. Loaded as a
trimesh Scene (not force="mesh") so all sub-meshes/materials/textures survive
the round trip; only a -90 deg rotation about X is applied ((x,y,z) ->
(x,z,-y): old up (Y) becomes new Z, old Z becomes new -Y).

Usage: python glb_yup_to_zup.py input.glb output.glb
"""
import sys

import numpy as np
import trimesh

ROTATE_YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python glb_yup_to_zup.py input.glb output.glb")
    in_glb, out_glb = sys.argv[1], sys.argv[2]

    scene = trimesh.load(in_glb, process=False)
    print(f"loaded {len(scene.geometry)} geometries, bounds (Y-up):\n{scene.bounds}")

    scene.apply_transform(ROTATE_YUP_TO_ZUP)
    print(f"bounds (Z-up):\n{scene.bounds}")

    scene.export(out_glb)
    print(f"wrote {out_glb}")


if __name__ == "__main__":
    main()
