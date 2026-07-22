"""OBJ -> GLB conversion without Blender.

Stand-in for mars-habitatsim/obj2glb.py (which needs bpy). No axis rotation:
habitat-sim treats stage meshes as Z-up (front +Y) and converts to its
internal Y-up world itself, so the GLB must keep hm2obj.py's Z-up
coordinates. (Verified empirically: a Y-up GLB renders standing vertically
in habitat; the Z-up one lies flat.)

Usage: python obj2glb_trimesh.py input.obj output.glb
"""
import sys

import trimesh


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python obj2glb_trimesh.py input.obj output.glb")
    in_obj, out_glb = sys.argv[1], sys.argv[2]

    mesh = trimesh.load(in_obj, force="mesh", process=False)
    mesh.export(out_glb)
    print(f"vertices {len(mesh.vertices)} faces {len(mesh.faces)}")
    print(f"bounds (m):\n{mesh.bounds}")
    print(f"wrote {out_glb}")


if __name__ == "__main__":
    main()
