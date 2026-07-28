"""Pure kinematic integration of Action into Pose.

Convention: yaw is the angle from the world +x axis toward +z, measured
counterclockwise in the x-z plane. v_fwd moves along the heading direction,
v_lat moves along the rightward perpendicular (yaw - 90deg). y (height) is
untouched here; it comes from terrain sampling elsewhere.
"""

import math

from sam_vla.core.types import Action, Pose


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def integrate_mars(pose: Pose, action: Action, dt: float) -> Pose:
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    dx = (action.v_fwd * cos_yaw + action.v_lat * sin_yaw) * dt
    dz = (action.v_fwd * sin_yaw - action.v_lat * cos_yaw) * dt

    new_yaw = _wrap_to_pi(pose.yaw + action.yaw_rate * dt)

    return Pose(x=pose.x + dx, y=pose.y, z=pose.z + dz, yaw=new_yaw)


if __name__ == "__main__":
    def _show(label: str, pose: Pose, action: Action, dt: float) -> None:
        result = integrate_mars(pose, action, dt)
        print(f"{label}:")
        print(f"  in  = {pose}, action={action}, dt={dt}")
        print(f"  out = {result}")
        print()

    # 1. Pure forward motion, facing +x (yaw=0): expect dx=v_fwd*dt, dz=0.
    _show(
        "forward, yaw=0",
        Pose(x=0.0, y=1.0, z=0.0, yaw=0.0),
        Action(v_fwd=2.0, v_lat=0.0, yaw_rate=0.0),
        dt=1.0,
    )

    # 2. Pure yaw rotation, no translation: expect x,z unchanged, yaw += pi/2.
    _show(
        "yaw rotation only",
        Pose(x=5.0, y=1.0, z=5.0, yaw=0.0),
        Action(v_fwd=0.0, v_lat=0.0, yaw_rate=math.pi / 2),
        dt=1.0,
    )

    # 3. Combined forward + lateral + yaw, facing +z (yaw=pi/2):
    #    forward should move along +z, lateral (right) should move along +x.
    _show(
        "combined, yaw=pi/2",
        Pose(x=0.0, y=1.0, z=0.0, yaw=math.pi / 2),
        Action(v_fwd=1.0, v_lat=1.0, yaw_rate=0.1),
        dt=0.5,
    )
