"""Generate a RoboDK program: full mold, then 1st plastic layer, then 1st carbon fiber layer on top.

Usage: open RobotPrint_Printhead.rdk in RoboDK, then run this script (RoboDK must be running).
Units: mm. The HORST KRL calls (horst_*, bosch_retract, reset_sps) only appear in the
exported .src file, not in simulation.

Expected files (searched next to the open RoboDK station, see resolve_data_dir):
  trajectories/layer_<i>/segment_<j>.csv   x,y,z,ux..wz,type   (plastic, 13 columns)
  fibers_transformed/layer_<i>/fiber_<i>_<j>.csv  x,y,z,nx,ny,nz   (fibers)
  mold_path.txt  x y z type layer  (optional mold, printed first)
Plastic and fibers must be expressed in the same frame (part frame).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

# ------------------------- PARAMETERS -------------------------
# Relative paths: RoboDK copies this script to a temp folder before running
# it, so neither __file__ nor the current directory point to the station
# folder. Data folders are resolved at runtime (see resolve_data_dir):
# the value as given, then next to the open RoboDK station
# (PATH_OPENSTATION), then the current directory.
TRAJECTORIES_DIR = "trajectories"
FIBERS_DIR = "fibers_transformed"
MOLD_FILE = "mold_path.txt"         # optional: printed first if present
PRINT_MOLD = True                   # False: skip the mold even if present

PLASTIC_LAYER = "first"             # "first" = first layer found, or a layer number
FIBER_LAYER = "first"               # "first" = smallest i in fibers_transformed/layer_i

FRAME_NAME = "PieceFrame"
FRAME_XYZRPY = [800, 0, 0, 0, 0, 0]  # ABSOLUTE in the station -> TO ADJUST
FORCE_FRAME_POSE = True

# --- Orientation ---
VERTICAL_SIGN = -1                  # plastic: tool Z towards -Z part (nozzle down)
FIBER_TOOL_SIGN = -1                # fibers: tool Z = FIBER_TOOL_SIGN * normal
FILE_ORIENTATION_SIGN = -1           # orientations read from the CSV files (piece): flip the
                                    # exported tool Z axis when it points the wrong way.
                                    # +1: keep as exported; -1: flip 180 deg about the
                                    # exported tool X axis (nozzle to the other side).
FIXED_X = [0, 1, 0]                 # reference vector for the tool's X axis

# --- Plastic ---
MIN_STEP = 3.0                      # mm: minimum distance between two kept points
DETOUR_Z_TOL = 0.2                  # mm: curve/connection point below print height - tol -> 'leave'
DETOUR_MAX_ZSPAN = 1.0              # mm: detour heuristic only valid on quasi-flat layers;
                                    # beyond this z spread over printed points the layer is
                                    # non-planar and detection is disabled
MAX_ANGLE = 5.0                     # deg: also keep a point if the orientation rotates more than this
DETOUR_XY_RADIUS = 1.0              # mm: local detour detection (non-planar layers): a print point
                                    # passing this close in XY under an already-laid line is a detour
TRAVEL_XY_RADIUS = 5.0              # mm: local travel lift (non-planar layers): approach/leave
                                    # points below a print line within this XY radius are raised
LIFT_TRAVEL = True                  # True: approach/leave points below the print height are
                                    # raised to zref + Z_LIFT
Z_LIFT = 3.0                        # mm: clearance above the print height for travel moves

# --- Speeds / rounding ---
SPEED_PLASTIC = 50                  # mm/s (0.05 m/s)
SPEED_TRAVEL = 80                    # mm/s (0.08 m/s)
ROUNDING = 0.5                      # mm, only between two consecutive print points

# --- Fibers ---
FIBER_MIN_STEP = 2.0                # mm
FIBER_MAX_ANGLE = 5.0               # deg
FIBER_ROUNDING = 0.5                # mm (0 on the cut point and the last point)
SPEED_FIBER = 5                     # mm/s (0.005 m/s) fiber laydown
SPEED_FIBER_NO_FIBRE = 20           # mm/s (0.02 m/s) too-short path: composite plastic without fiber
MINIMAL_COMPOSITE_LENGTH = 80       # mm: below this, no fiber (as in the postprocessor)
DISTANCE_CUT_TO_END = 50            # mm: fiber cut this distance before the end of the path
FIBER_HOP = 5                       # mm: retreat along the nozzle between two fibers
LIFT_Z = 10                         # mm: vertical lift after the mold/plastic, before switching

# --- HORST KRL calls (per the HORST robotic printing API) ---
TP_TEMP, CR_TEMP, CF_TEMP = 230, 205, 200
# Initialization block (BAS): emitted as inserted code at the very start
INIT_BAS_CODE = [
    "BAS (#INITMOV,0)",
    "BAS (#VEL_PTP,100)",
    "BAS (#ACC_PTP,20)",
    "$VEL.CP=0.04",
    "BAS (#TOOL,0)",
    "BAS (#BASE,0)",
]
# Printhead preparation: init -> extend -> temp -> set_feed
HEAD_INIT_CODE = [
    "horst_init()",
    "horst_extend(1000, 0, 0, 1000)",
    f"horst_temp(TRUE, {TP_TEMP}, TRUE, {CR_TEMP}, TRUE, {CF_TEMP})",
]
PLASTIC_ON = "horst_set_feed(35, 0)"        # TP feed 35%, no fiber feed
FIBER_ON = "horst_set_feed(13, 100)"        # TP feed + CF feed 100%
FIBER_ON_NO_FIBRE = "horst_set_feed(13, 0)"  # too-short path: composite plastic, no fiber
FEED_OFF = "horst_set_feed(0, 0)"
CUT_CODE = ["horst_cut()"]
FIBER_RETRACT = "horst_retract(0, 200, 1000)"   # retract CF material before the cut
# Termination block: cut -> retract modules -> heaters off -> PLC reset
TERMINATION_CODE = [
    "horst_cut()",
    "bosch_retract()",
    "horst_temp(FALSE, 0, FALSE, 0, FALSE, 0)",
    "reset_sps()",
]

CHECK_REACH = True
CHECK_STEP = 2
DRAW_CURVES = True                  # mold orange, plastic red, fibers blue (RoboDK curves)
# --------------------------------------------------------------

PRINT_TYPES = ("curve", "connection")


# ----------------------------- data -----------------------------
def _layer_zref(P, types, layers):
    """Dominant print height per layer (mode of the rounded z of print points)."""
    zref = {}
    for li in np.unique(layers):
        m = (layers == li) & np.isin(types, PRINT_TYPES)
        if m.any():
            z = np.round(P[m][:, 2], 3)
            zref[li] = float(np.bincount(np.round((z - z.min()) * 1000).astype(int)).argmax()) / 1000 + z.min()
    return zref


def relabel_detours_xy(P, types, xy_radius=DETOUR_XY_RADIUS, z_tol=DETOUR_Z_TOL):
    """Local detour detection for non-planar layers: a curve/connection point
    passing under an already-laid line (near-coincident in XY, lower in z)
    is relabeled 'leave': not printed. Replaces the per-layer print-height
    heuristic, which is meaningless when z varies across a layer."""
    m = np.isin(types, PRINT_TYPES)
    idx = np.where(m)[0]
    detour = np.zeros(len(P), bool)
    if len(idx) >= 2:
        tree = cKDTree(P[idx, :2])
        pairs = tree.query_pairs(xy_radius, output_type="ndarray")
        if len(pairs):
            a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
            za, zb = P[a, 2], P[b, 2]
            detour[a[za < zb - z_tol]] = True
            detour[b[zb < za - z_tol]] = True
    types[detour] = "leave"
    return int(detour.sum())


def lift_travel_xy(P, types, xy_radius=TRAVEL_XY_RADIUS, z_lift=Z_LIFT, z_tol=DETOUR_Z_TOL):
    """Local travel lift for non-planar layers: approach/leave points below a
    nearby print line are raised above it (clearance for travel moves)."""
    m = np.isin(types, PRINT_TYPES)
    lifted = 0
    if m.any():
        print_pts = P[m]
        tree = cKDTree(print_pts[:, :2])
        for i in np.where(np.isin(types, ("approach", "leave")))[0]:
            nb = tree.query_ball_point(P[i, :2], xy_radius)
            if nb:
                z_local = print_pts[nb, 2].max()
                if P[i, 2] < z_local - z_tol:
                    P[i, 2] = z_local + z_lift
                    lifted += 1
    return lifted


def relabel_detours(P, types, layers, z_tol=DETOUR_Z_TOL):
    """curve/connection points below their layer's print height (passing under
    an already-laid line) are relabeled 'leave': not printed."""
    zref = _layer_zref(P, types, layers)
    detour = np.zeros(len(P), bool)
    for li, zr in zref.items():
        m = (layers == li) & np.isin(types, PRINT_TYPES)
        detour |= m & (P[:, 2] < zr - z_tol)
    types[detour] = "leave"
    return int(detour.sum())


def lift_travel(P, types, layers, z_lift=Z_LIFT, z_tol=DETOUR_Z_TOL):
    """approach/leave points below their layer's print height are raised to
    zref + z_lift (clearance above the print plane for travel moves)."""
    zref = _layer_zref(P, types, layers)
    lifted = 0
    for li, zr in zref.items():
        m = (layers == li) & np.isin(types, ("approach", "leave")) & (P[:, 2] < zr - z_tol)
        P[m, 2] = zr + z_lift
        lifted += int(m.sum())
    return lifted


@dataclass
class PlasticPath:
    points: np.ndarray           # (N, 3)
    types: np.ndarray            # (N,) strings: curve / connection / approach / leave
    layers: np.ndarray           # (N,) layer id per point (mold: read from file; piece: constant)
    n_detours: int = 0
    source: str = ""
    orientations: np.ndarray | None = None   # (N, 3, 3) tool rotations exported by the slicer


@dataclass
class Move:
    pose: np.ndarray             # 4x4
    speed: float
    rounding: float
    label: str
    pre: list = field(default_factory=list)     # ("comment"|"code"|"pause", value)
    post: list = field(default_factory=list)


def read_csv(path):
    """Read a CSV with optional header -> (rows, header|None).

    Returns raw string rows: the type column can be a label
    ("curve", ...) or a numeric value, depending on the export."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    try:
        float(rows[0][0])
        return rows, None
    except ValueError:
        return rows[1:], rows[0]


def resolve_data_dir(path, extra_bases=()):
    """Find the data folder: the path as given, then the same folder name
    next to the open RoboDK station / in the given directories."""
    candidates = [path]
    name = os.path.basename(path.rstrip("/\\"))
    for base in extra_bases:
        if base:
            candidates.append(os.path.join(base, name))
    for c in candidates:
        if os.path.isdir(c):
            return c
    return path


def resolve_data_file(path, extra_bases=()):
    """Find a data file: as given, then next to the open RoboDK station / in
    the given directories (same file name)."""
    candidates = [path]
    name = os.path.basename(path)
    for base in extra_bases:
        if base:
            candidates.append(os.path.join(base, name))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return path


# ----------------------- geometry helpers -----------------------
def frames_from_z(P, Z, ref):
    """4x4 matrices: given Z, X = projection of ref onto the plane normal to Z."""
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    ref = np.tile(np.array(ref, float), (len(Z), 1))
    X = ref - (ref * Z).sum(1, keepdims=True) * Z
    bad = np.linalg.norm(X, axis=1) < 0.1          # Z nearly parallel to ref: fallback
    if bad.any():
        alt = np.tile(np.array([1.0, 0, 0]) if abs(ref[0, 0]) < 0.9 else np.array([0, 1.0, 0]), (bad.sum(), 1))
        X[bad] = alt - (alt * Z[bad]).sum(1, keepdims=True) * Z[bad]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.cross(Z, X)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = X, Y, Z, P
    return T


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def decimate_path(T, step, max_angle, forced=()):
    """Kept indices: start, end, forced points, and any point > step or > max_angle
    from the last kept one."""
    keep, last = [0], 0
    forced = set(forced)
    for i in range(1, len(T)):
        if (i in forced or np.linalg.norm(T[i, :3, 3] - T[last, :3, 3]) >= step
                or rot_angle(T[last, :3, :3], T[i, :3, :3]) >= max_angle):
            keep.append(i)
            last = i
    if keep[-1] != len(T) - 1:
        keep.append(len(T) - 1)
    return keep


# ----------------------------- mold -----------------------------
def load_mold(path):
    """Load the mold trajectory from the single whitespace-separated txt
    export: 'x y z type layer' per row, with the segment type as a text
    label (see SlicedMold.export_layers_to_txt)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue  # header line
            layer = int(float(parts[4])) if len(parts) >= 5 else 0
            rows.append((x, y, z, parts[3], layer))
    if not rows:
        raise SystemExit(f"{path}: no usable rows found")
    P = np.array([r[0:3] for r in rows])
    types = np.array([r[3] for r in rows], dtype=object)
    layers = np.array([r[4] for r in rows])

    # Detour detection per layer (a mold layer is quasi-flat by construction).
    ndet = relabel_detours(P, types, layers)
    if LIFT_TRAVEL:
        nlift = lift_travel(P, types, layers)
    else:
        nlift = 0

    # consecutive exact duplicates
    keep = (np.abs(np.diff(P, axis=0)).sum(1) > 1e-6) | (types[1:] != types[:-1])
    keep = np.r_[True, keep]
    P, types, layers = P[keep], types[keep], layers[keep]
    path_obj = PlasticPath(points=P, types=types, layers=layers,
                           n_detours=ndet, source=path)
    print(f"Mold: {len(P)} points from {os.path.basename(path)}, "
          f"layers {layers.min()}..{layers.max()}, {ndet} detour points -> leave, {nlift} lifted")
    return path_obj


# ----------------------------- plastic -----------------------------
def load_plastic(traj_dir=TRAJECTORIES_DIR, layer=PLASTIC_LAYER, prefix="plastic "):
    """Load all segment_<j>.csv files of the requested layer into one continuous path."""
    layer_dirs = {}
    for p in glob.glob(os.path.join(traj_dir, "layer_*")):
        m = re.fullmatch(r"layer_(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            layer_dirs[int(m.group(1))] = p
    if not layer_dirs:
        raise SystemExit(f"No layer_<i> folder in {traj_dir}")
    li = min(layer_dirs) if layer == "first" else int(layer)
    if li not in layer_dirs:
        raise SystemExit(f"Plastic layer {li} not found (available: {sorted(layer_dirs)})")

    files = []
    for p in glob.glob(os.path.join(layer_dirs[li], "segment_*.csv")):
        m = re.fullmatch(r"segment_(\d+)\.csv", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if not files:
        raise SystemExit(f"No segment_<j>.csv in {layer_dirs[li]}")

    points, types, rots = [], [], []
    for _, path in files:
        rows, header = read_csv(path)
        if not rows or len(rows[0]) < 4:
            raise SystemExit(f"{path}: expected x,y,z[,axes],type (at least 4 columns)")
        pts = np.array([[float(v) for v in r[0:3]] for r in rows])
        # Full tool orientation exported by the slicer (13 columns:
        # x,y,z then the 9 coefficients of the X/Y/Z axes), else None.
        R = None
        if len(rows[0]) >= 12:
            try:
                # The 9 coefficients are the tool's X, Y, Z axes in a row:
                # the rotation is rebuilt by transposing (columns = axes,
                # see build_orthonormal_frames).
                R = np.array([[float(v) for v in r[3:12]] for r in rows])
                R = R.reshape(len(rows), 3, 3).transpose(0, 2, 1)
            except ValueError:
                R = None
        # Type column: last column, a text label (curve, ...) or a numeric
        # value (0 = leave, 1 = curve).
        raw = [r[-1] for r in rows]
        try:
            float(raw[0])
            labels = ["curve" if float(v) > 0.5 else "leave" for v in raw]
        except ValueError:
            labels = raw
        points.append(pts)
        types.append(np.array(labels, dtype=object))
        rots.append(R)
    P = np.vstack(points)
    T = np.concatenate(types)
    R = np.vstack([r for r in rots if r is not None]) if all(r is not None for r in rots) else None
    if len(P) == 0:
        raise SystemExit(f"No points in {layer_dirs[li]}")

    # Detours (below an already-laid line) -> 'leave': no printing.
    # Heuristic based on the dominant print height per layer: only valid
    # on a quasi-flat layer; on a non-planar layer (z spread over more
    # than DETOUR_MAX_ZSPAN) a lower z is legitimate, so detect nothing.
    is_print = np.isin(T, PRINT_TYPES)
    layers_arr = np.full(len(P), li)
    ndet = 0
    if is_print.any():
        z_print = P[is_print][:, 2]
        if z_print.max() - z_print.min() <= DETOUR_MAX_ZSPAN:
            ndet = relabel_detours(P, T, layers_arr)
            if LIFT_TRAVEL:
                lift_travel(P, T, layers_arr)
        else:
            # Non-planar layer: the per-layer print-height heuristic is
            # meaningless; use local XY detection instead (a point passing
            # under an already-laid line is a detour).
            print(f"  non-planar layer (z spread {z_print.max() - z_print.min():.1f} mm): "
                  "local (XY) detour detection")
            ndet = relabel_detours_xy(P, T)
            if LIFT_TRAVEL:
                lift_travel_xy(P, T)

    # consecutive exact duplicates
    keep = (np.abs(np.diff(P, axis=0)).sum(1) > 1e-6) | (T[1:] != T[:-1])
    keep = np.r_[True, keep]
    P, T = P[keep], T[keep]
    if R is not None:
        R = R[keep]

    path = PlasticPath(points=P, types=T, layers=layers_arr, n_detours=ndet,
                       source=layer_dirs[li], orientations=R)
    print(f"{prefix.capitalize()}: layer {li}, {len(files)} CSV segments, {len(P)} points, "
          f"{ndet} detour points -> leave")
    return path


def plastic_moves(p: PlasticPath, prefix="plastic "):
    # TCP poses: full orientations exported by the slicer when present,
    # otherwise vertical Z (nozzle down) + reference X.
    if p.orientations is not None:
        T = np.tile(np.eye(4), (len(p.points), 1, 1))
        R = p.orientations
        if FILE_ORIENTATION_SIGN < 0:
            # Flip the tool 180 deg about its own X axis: Z -> -Z, Y -> -Y.
            # F acts on the world side (left multiplication) so the exported
            # tool X (direction of travel) is preserved.
            flip = np.diag([1.0, -1.0, -1.0])
            R = flip @ R
        T[:, :3, :3] = R
        T[:, :3, 3] = p.points
    else:
        Z = np.tile([0.0, 0.0, float(VERTICAL_SIGN)], (len(p.points), 1))
        T = frames_from_z(p.points, Z, FIXED_X)
    types = p.types
    layers = p.layers

    # Decimation: keep points on segment (type or layer) changes, and any
    # point further than MIN_STEP or rotated more than MAX_ANGLE from the
    # last kept one (same rules as the original mold viewer).
    keep, last = [0], 0
    for i in range(1, len(T)):
        seg_change = types[i] != types[i - 1] or layers[i] != layers[i - 1]
        dist = np.linalg.norm(T[i, :3, 3] - T[last, :3, 3])
        ang = rot_angle(T[last, :3, :3], T[i, :3, :3])
        if seg_change:
            if keep[-1] != i - 1:
                keep.append(i - 1)
            keep.append(i)
            last = i
        elif dist >= MIN_STEP or ang >= MAX_ANGLE:
            keep.append(i)
            last = i
    if keep[-1] != len(T) - 1:
        keep.append(len(T) - 1)
    idx = sorted(set(keep))

    moves, prev_type, printing_prev = [], None, False
    for n, i in enumerate(idx):
        t = types[i]
        printing = t in PRINT_TYPES
        nxt = types[idx[n + 1]] if n + 1 < len(idx) else None
        pre = []
        if t != prev_type:
            pre.append(("comment", f"{prefix}layer {layers[i]} - {t}"))
            prev_type = t
        if printing and not printing_prev:
            pre.append(("code", PLASTIC_ON))
        if not printing and printing_prev:
            pre.append(("code", FEED_OFF))
        printing_prev = printing
        moves.append(Move(
            pose=T[i],
            speed=SPEED_PLASTIC if t not in ("approach", "leave") else SPEED_TRAVEL,
            rounding=ROUNDING if (t == "curve" and nxt in PRINT_TYPES) else 0,
            label=f"{prefix}{t} x={p.points[i,0]:.1f} y={p.points[i,1]:.1f} z={p.points[i,2]:.1f}",
            pre=pre, post=[]))
    if printing_prev:
        moves[-1].post.append(("code", FEED_OFF))

    # Print polylines for the trace, one per continuous run of print points
    # within the same layer; the preceding approach point (at print height)
    # is prepended so a run whose only point is 'curve' still shows up
    # (same behaviour as the original mold viewer).
    runs, cur = [], []
    is_p = np.isin(types, PRINT_TYPES)
    for i in range(len(p.points)):
        if is_p[i] and (not cur or layers[i] == layers[cur[-1]]):
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i] if is_p[i] else []
    if cur:
        runs.append(cur)
    for run in runs:
        j = run[0] - 1
        if j >= 0 and types[j] == "approach" and layers[j] == layers[run[0]]:
            run.insert(0, j)
    runs = [np.array([p.points[i] for i in run]) for run in runs if len(run) > 1]
    print(f"{prefix.capitalize()}: {len(p.points)} points -> {len(moves)} moves, {len(runs)} trace runs")
    return moves, runs


# ------------------------------- fibers -------------------------------
def load_fiber(path):
    rows, _ = read_csv(path)
    try:
        A = np.array([[float(v) for v in r] for r in rows], ndmin=2)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")
    if A.ndim != 2 or A.shape[1] < 6:
        raise SystemExit(f"{path}: expected x,y,z,nx,ny,nz (6 columns)")
    P, N = A[:, 0:3], A[:, 3:6]
    keep = np.r_[True, np.abs(np.diff(P, axis=0)).sum(1) > 1e-6]   # consecutive duplicates
    return P[keep], N[keep]


def find_fiber_files(fiber_dir=FIBERS_DIR, layer=FIBER_LAYER):
    layers = {}
    for p in glob.glob(os.path.join(fiber_dir, "layer_*")):
        m = re.fullmatch(r"layer_(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            layers[int(m.group(1))] = p
    if not layers:
        raise SystemExit(f"No layer_<i> folder in {fiber_dir}")
    li = min(layers) if layer == "first" else int(layer)
    if li not in layers:
        raise SystemExit(f"Fiber layer {li} not found (available: {sorted(layers)})")
    files = []
    for p in glob.glob(os.path.join(layers[li], f"fiber_{li}_*.csv")):
        m = re.fullmatch(rf"fiber_{li}_(\d+)\.csv", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if not files:
        raise SystemExit(f"No fiber_{li}_<j>.csv file in {layers[li]}")
    print(f"Fibers: layer {li}, {len(files)} fibers")
    return li, [p for _, p in files]


def fiber_moves(files):
    moves, polylines = [], []
    for path in files:
        P, N = load_fiber(path)
        if len(P) < 2:
            print("  fiber skipped (fewer than 2 points):", os.path.basename(path))
            continue
        polylines.append(P)
        T = frames_from_z(P, FIBER_TOOL_SIGN * N, FIXED_X)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        L = s[-1]
        with_fibre = L >= MINIMAL_COMPOSITE_LENGTH
        cut_i = None
        if with_fibre:
            below = np.where(s < L - DISTANCE_CUT_TO_END)[0]
            cut_i = int(below[-1]) if len(below) else None
        idx = decimate_path(T, FIBER_MIN_STEP, FIBER_MAX_ANGLE, forced=[cut_i] if cut_i is not None else [])
        name = os.path.basename(path)

        def hop(Tk):
            H = Tk.copy()
            H[:3, 3] = Tk[:3, 3] - FIBER_HOP * Tk[:3, 2]       # retreat along -Z of the TCP
            return H

        # 1) retreat above the start
        moves.append(Move(
            pose=hop(T[0]), speed=SPEED_TRAVEL, rounding=0,
            pre=[("comment", f"{name} - {L:.0f} mm, {'fiber' if with_fibre else 'NO fiber (too short)'}")],
            label=f"{name} start retreat"))
        # 2) fiber start; extrusion starts once arrived
        speed = SPEED_FIBER if with_fibre else SPEED_FIBER_NO_FIBRE
        moves.append(Move(
            pose=T[idx[0]], speed=SPEED_TRAVEL, rounding=0,
            post=[("code", FIBER_ON if with_fibre else FIBER_ON_NO_FIBRE)],
            label=f"{name} start"))
        # 3) laydown
        for k, i in enumerate(idx[1:], start=1):
            last = (k == len(idx) - 1)
            is_cut = (i == cut_i)
            post = []
            if is_cut:
                post += [("code", FIBER_RETRACT)] + [("code", c) for c in CUT_CODE]
            if last:
                post += [("code", FEED_OFF)]
            moves.append(Move(
                pose=T[i], speed=speed,
                rounding=0 if (last or is_cut) else FIBER_ROUNDING,
                label=f"{name} pt {i}", post=post))
        # 4) retreat after the end
        moves.append(Move(pose=hop(T[idx[-1]]), speed=SPEED_TRAVEL, rounding=0,
                          label=f"{name} end retreat"))
        print(f"  {name}: {len(P)} points -> {len(idx)} kept, length {L:.0f} mm"
              f"{'' if with_fibre else ' (no fiber)'}"
              f"{f', cut at point {cut_i}' if cut_i is not None else ''}")
    return moves, polylines


# ------------------------------- RoboDK -------------------------------
def draw_curves(RDK, robolink, frame, polylines, prefix, color):
    for it in RDK.ItemList(robolink.ITEM_TYPE_OBJECT, False):
        if it.Name().startswith(prefix):
            it.Delete()
    F = np.array(frame.PoseAbs().rows)
    for k, pts in enumerate(polylines):
        pts_abs = (F[:3, :3] @ pts.T).T + F[:3, 3]
        c = RDK.AddCurve(pts_abs.tolist())
        c.setName(f"{prefix} {k + 1}")
        c.setColor(color)


def main(argv=None):
    from robodk import robolink, robomath

    args = parse_args(argv)

    RDK = robolink.Robolink()
    # Open station folder: the data (trajectories/, fibers_transformed/,
    # mold_path.txt) lives next to the .rdk file.
    station_dir = RDK.getParam("PATH_OPENSTATION") or ""
    bases = [station_dir, os.getcwd()]
    traj_dir = resolve_data_dir(args.trajectories, bases)
    fibers_dir = resolve_data_dir(args.fibers, bases)
    print(f"Station folder: {station_dir or '?'}")
    print(f"Trajectories: {traj_dir}\nFibers: {fibers_dir}")

    # --- mold (optional): printed first, plastic only ---
    mold_moves, mold_runs = [], []
    if PRINT_MOLD and not args.no_mold:
        mold_file = resolve_data_file(args.mold, bases)
        if os.path.isfile(mold_file):
            mold = load_mold(mold_file)
            mold_moves, mold_runs = plastic_moves(mold, prefix="mold ")
            print(f"Mold: {len(mold_moves)} moves")
        else:
            print(f"Mold skipped: file {mold_file} not found")

    plastic = load_plastic(traj_dir, args.plastic_layer)
    moves, plastic_runs = plastic_moves(plastic)

    _, files = find_fiber_files(fibers_dir, args.fiber_layer)
    fmoves, fiber_lines = fiber_moves(files)

    # mold first, then a lift before the part
    if mold_moves:
        lift = mold_moves[-1].pose.copy()
        lift[2, 3] += LIFT_Z
        mold_moves.append(Move(
            pose=lift, speed=SPEED_TRAVEL, rounding=0,
            pre=[("comment", "mold done: lift before the part")],
            label="lift after mold"))
    moves = mold_moves + moves

    # vertical lift after the plastic, then extruder change
    lift = moves[-1].pose.copy()
    lift[2, 3] += LIFT_Z
    moves.append(Move(
        pose=lift, speed=SPEED_TRAVEL, rounding=0,
        pre=[("comment", "extruder change: plastic -> composite")],
        label="lift before fibers"))
    moves += fmoves

    # termination: heaters off etc., emitted after the last move
    moves[-1].post += [("code", c) for c in TERMINATION_CODE]

    robot = RDK.ItemUserPick("Pick the robot", robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise SystemExit("No robot picked")
    tool = robot.getLink(robolink.ITEM_TYPE_TOOL)
    if not tool.Valid():
        raise SystemExit("No tool attached to the robot: attach the printhead in RoboDK")

    frame = RDK.Item(FRAME_NAME, robolink.ITEM_TYPE_FRAME)
    if not frame.Valid():
        frame = RDK.AddFrame(FRAME_NAME)
        force = True
        print(f"WARNING: no frame named '{FRAME_NAME}' found -> created at FRAME_XYZRPY. "
              f"If you moved a frame manually, check its exact name.")
    else:
        force = FORCE_FRAME_POSE
    if force:
        frame.setParent(RDK.ActiveStation())
        frame.setPoseAbs(robomath.xyzrpw_2_pose(FRAME_XYZRPY))
    print("Part frame (absolute):", np.round(frame.PoseAbs().Pos(), 1),
          "| robot base (absolute):", np.round(robot.PoseAbs().Pos(), 1))

    old = RDK.Item("PrintPlasticFiber", robolink.ITEM_TYPE_PROGRAM)
    if old.Valid():
        old.Delete()

    def mat(M):
        return robomath.Mat(M.tolist())

    robot.setPoseFrame(frame)
    robot.setPoseTool(tool)

    if CHECK_REACH:
        tested = list(range(0, len(moves), CHECK_STEP))
        bad = [k for k in tested
               if len(robot.SolveIK(mat(moves[k].pose)).rows) < 6]
        print(f"IK test ({len(tested)} moves tested): {len(bad)} unreachable")
        if bad:
            F = np.array(frame.PoseAbs().rows)          # part frame -> station
            base = np.array(robot.PoseAbs().Pos())       # robot base in station
            for k in bad[:8]:
                p_abs = F[:3, :3] @ moves[k].pose[:3, 3] + F[:3, 3]
                dist = np.linalg.norm(p_abs - base)
                print(f"  move {k + 1}: {moves[k].label}")
                print(f"      in part frame : {np.round(moves[k].pose[:3, 3], 1).tolist()}"
                      f" | absolute: {np.round(p_abs, 1).tolist()}"
                      f" | {dist:.0f} mm from robot base")

    if DRAW_CURVES:
        draw_curves(RDK, robolink, frame, mold_runs, "MoldTrace", [1, 0.6, 0, 1])
        draw_curves(RDK, robolink, frame, plastic_runs, "PrintTrace", [1, 0, 0, 1])
        draw_curves(RDK, robolink, frame, fiber_lines, "FiberTrace", [0, 0.3, 1, 1])

    prog = RDK.AddProgram("PrintPlasticFiber")
    prog.setPoseFrame(frame)
    prog.setPoseTool(tool)

    def emit(items):
        for kind, val in items:
            if kind == "comment":
                prog.RunInstruction(val, robolink.INSTRUCTION_COMMENT)
            elif kind == "code":
                prog.RunInstruction(val, robolink.INSTRUCTION_INSERT_CODE)
            elif kind == "pause":
                prog.Pause(val)

    # Initialization: BAS block + printhead preparation, before any movement
    emit([("comment", "--- Initialization ---")])
    emit([("code", c) for c in INIT_BAS_CODE])
    emit([("comment", "--- Printhead preparation ---")])
    emit([("code", c) for c in HEAD_INIT_CODE])

    cur_speed = cur_round = None
    for n, m in enumerate(moves):
        if m.speed != cur_speed:
            prog.setSpeed(m.speed)
            cur_speed = m.speed
        if m.rounding != cur_round:
            prog.setRounding(m.rounding)
            cur_round = m.rounding
        emit(m.pre)
        if n == 0:
            prog.MoveJ(mat(m.pose))
        else:
            prog.MoveL(mat(m.pose))
        emit(m.post)

    try:
        print("RoboDK check:", prog.Update()[-1])
    except Exception as e:
        print("prog.Update() unavailable:", e)
    print(f"Program 'PrintPlasticFiber' created: {len(moves)} moves. Run it or simulate it.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", default=TRAJECTORIES_DIR,
                        help="folder holding layer_<i>/segment_<j>.csv (plastic)")
    parser.add_argument("--fibers", default=FIBERS_DIR,
                        help="folder holding layer_<i>/fiber_<i>_<j>.csv (fibers)")
    parser.add_argument("--plastic-layer", default=PLASTIC_LAYER,
                        help="'first' or a plastic layer number")
    parser.add_argument("--fiber-layer", default=FIBER_LAYER,
                        help="'first' or a fiber layer number")
    parser.add_argument("--mold", default=MOLD_FILE,
                        help="mold txt file (empty = skip the mold)")
    parser.add_argument("--no-mold", action="store_true",
                        help="do not print the mold even if present")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
