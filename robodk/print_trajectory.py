"""
RoboDK trajectory loader for the slicer's exported path files.

Reads a trajectory file whose rows are only:
    x y z [nx ny nz] type [layer]
(no tool axes, no travel tangent exported). The robot orientation is
reconstructed here, point by point:

- The tool's Z axis is aligned with the local surface normal (flipped to
  point INTO the surface, the usual approach convention; see
  APPROACH_SIGN below).
- The tool's X axis is the local travel direction (tangent), re-orthogonalized
  against Z so (X, Y, Z) is always a valid right-handed basis.
- Y = Z x X.

The tangent is computed from the trajectory points themselves: a central
difference for interior points, one-sided at segment endpoints. On
approach/leave moves (travel along the normal itself, tangent undefined) the
X axis is kept continuous with the neighbouring printed point instead of
using an arbitrary fallback, so the tool does not spin about its Z axis
between the approach move and the start of the curve.

Setup:
- A RoboDK station with a robot, a tool frame (TOOL_NAME) and a reference
  frame (FRAME_NAME) matching the slicer's world coordinates.
- Run once: python print_trajectory.py <path_file>
  or copy this file into a RoboDK "Python program" and set PATH_FILE below.
"""

import sys

from robodk.robomath import Mat, cross, norm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

RDK = None
try:
    from robolink import Robolink

    RDK = Robolink()
except ImportError:
    pass

ROBOT_NAME = "robot"
TOOL_NAME = "End Effector"      # tool frame in the station
FRAME_NAME = "Piece Frame"      # reference frame matching the slicer world
PATH_FILE = "trajectories/piece_path.txt"

# -1: tool Z axis points INTO the surface (usual deposition convention)
# +1: tool Z axis points AWAY from the surface
APPROACH_SIGN = -1.0


# ----------------------------------------------------------------------------
# Vector helpers (robodk.robomath also has these; kept explicit for clarity)
# ----------------------------------------------------------------------------

def unit(v):
    n = norm(v)
    if n < 1e-12:
        return [0.0, 0.0, 1.0]
    return [v[0] / n, v[1] / n, v[2] / n]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def tangent_at(points, i):
    """Travel direction at points[i]: central difference inside the
    segment, one-sided difference at the two endpoints."""
    if len(points) == 1:
        return None
    if i == 0:
        return sub(points[1], points[0])
    if i == len(points) - 1:
        return sub(points[-1], points[-2])
    return sub(points[i + 1], points[i - 1])


def build_frame(z_pref, x_pref):
    z = unit(z_pref)

    x = list(x_pref)
    dot = sum(c * d for c, d in zip(unit(x) if norm(x) > 1e-12 else x, z))
    x = [c - dot * d for c, d in zip(x, z)]
    if norm(x) < 1e-8:
        # travel direction along the normal (approach/leave): keep the
        # previous X projected perpendicular to Z, or a global fallback.
        if build_frame.last_x is not None:
            x = list(build_frame.last_x)
            dot = sum(c * d for c, d in zip(x, z))
            x = [c - dot * d for c, d in zip(x, z)]
        if norm(x) < 1e-8:
            x = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
            dot = sum(c * d for c, d in zip(x, z))
            x = [c - dot * d for c, d in zip(x, z)]
    x = unit(x)
    y = cross(z, x)

    return x, y, z


build_frame.last_x = None


def frame_to_pose(x, y, z, p):
    """(X, Y, Z) axes + position -> 4x4 pose as a robodk Mat."""
    return Mat(
        [
            [x[0], y[0], z[0], p[0]],
            [x[1], y[1], z[1], p[1]],
            [x[2], y[2], z[2], p[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# ----------------------------------------------------------------------------
# Trajectory file parsing
# ----------------------------------------------------------------------------

def parse_path_file(filepath, include_normal=True):
    """Return (points, normals, types): one entry per row, in file order.
    normals[i] is None when the file has no normal columns."""
    points, normals, types = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] in ("x",):  # header
                continue
            if include_normal:
                x, y, z, nx, ny, nz, seg_type = parts[:7]
                normals.append([float(nx), float(ny), float(nz)])
            else:
                x, y, z, seg_type = parts[:4]
            points.append([float(x), float(y), float(z)])
            types.append(seg_type)
    return points, normals, types


# ----------------------------------------------------------------------------
# Program generation
# ----------------------------------------------------------------------------

def generate_program(rdk, robot, tool, frame, filepath, include_normal=True,
                     approach_sign=-1.0):
    points, normals, types = parse_path_file(filepath, include_normal)

    if not points:
        raise ValueError("Empty trajectory file: %s" % filepath)

    # Group consecutive rows into segments of a common type, so the
    # tangent is never computed across an approach/leave boundary.
    segments = []
    start = 0
    for i in range(1, len(points) + 1):
        if i == len(points) or types[i] != types[start]:
            segments.append((types[start], points[start:i],
                             normals[start:i] if include_normal else None))
            start = i

    program = rdk.AddProgram("PrintTrajectory")
    program.setRobot(robot)
    program.setTool(tool)
    program.setFrame(frame)

    build_frame.last_x = None
    n_moves = 0

    for seg_type, seg_points, seg_normals in segments:
        for i, p in enumerate(seg_points):
            if include_normal:
                n = seg_normals[i]
            else:
                n = sub(p, seg_points[i - 1]) if i > 0 else (
                    sub(seg_points[i + 1], p) if i + 1 < len(seg_points) else [0, 0, 1.0])
            z_pref = [approach_sign * c for c in unit(n)]
            x_pref = tangent_at(seg_points, i)

            if x_pref is None:
                # single-point segment: keep continuity with previous frame
                x_pref = build_frame.last_x or [1.0, 0.0, 0.0]

            x, y, z = build_frame(z_pref, x_pref)
            build_frame.last_x = x

            pose = frame_to_pose(x, y, z, p)

            if seg_type in ("approach", "leave"):
                program.MoveJ(pose)
            else:
                program.MoveL(pose)
            n_moves += 1

    return program, n_moves


def main():
    global PATH_FILE
    if len(sys.argv) > 1:
        PATH_FILE = sys.argv[1]

    if RDK is None:
        raise SystemExit(
            "robolink/robodk not available: run this script from RoboDK's "
            "Python environment (or install the robodk package)."
        )

    robot = RDK.Item(ROBOT_NAME)
    tool = RDK.Item(TOOL_NAME)
    frame = RDK.Item(FRAME_NAME)

    program, n_moves = generate_program(
        RDK, robot, tool, frame, PATH_FILE,
        include_normal=True,
        approach_sign=APPROACH_SIGN,
    )
    print("Program generated: %s (%d moves)" % (program.Name(), n_moves))


if __name__ == "__main__":
    main()
