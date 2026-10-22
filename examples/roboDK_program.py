"""Generate a RoboDK program: full mold, then 1st plastic layer, then 1st carbon fiber layer on top.

Usage: open RobotPrint_Printhead.rdk in RoboDK, then run this script (RoboDK must be running).
Units: mm. The HORST KRL calls (horst_*, bosch_retract, reset_sps) only appear in the
exported .src file, not in simulation.

Expected files (searched next to the open RoboDK station, see resolve_data_dir):
  piece_path.csv    x,y,z,nx,ny,nz,type,layer  (plastic piece, new normal-only export)
  mold_path.txt     x y z nx ny nz type layer   (optional mold, printed first)
  fibers_transformed/layer_<i>/fiber_<i>_<j>.csv  x,y,z,nx,ny,nz   (fibers)
Piece and fibers must be expressed in the same frame (part frame).

The trajectory files no longer contain any tool axes: only the points, their
local surface normal and the segment type. The tool orientation is rebuilt
here from the normal ONLY: TCP Z = PIECE_TOOL_SIGN * normal. No travel
tangent is used: the rotation about the nozzle axis is kept as constant as
possible (TCP X = projection of a fixed reference vector, see FIXED_X),
which avoids the tool spinning about its nozzle and the IK/simulation
errors that come with it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

# ------------------------- PARAMETERS -------------------------
# Relative paths: RoboDK copies this script to a temp folder before running
# it, so neither __file__ nor the current directory point to the station
# folder. Data folders are resolved at runtime (see resolve_data_dir):
# the value as given, then next to the open RoboDK station
# (PATH_OPENSTATION), then the current directory.
FIBERS_DIR = "fibers_transformed"
PIECE_FILE = "piece_path.csv"        # plastic piece: printed after the mold
MOLD_FILE = "mold_path.txt"         # optional: printed first if present
PRINT_MOLD = True                   # False: skip the mold even if present

PLASTIC_LAYERS = "first"           # "first" = first layer found; "all" = every layer in
                                    # order; or a list of layer numbers, e.g. [0, 1]
FIBER_LAYER = "first"               # "first" = smallest i in fibers_transformed/layer_i

FRAME_NAME = "PieceFrame"
FRAME_XYZRPY = [800, 0, 0, 0, 0, 0]  # ABSOLUTE in the station -> TO ADJUST
FORCE_FRAME_POSE = True

# --- Orientation ---
VERTICAL_TOOL = True           # MOLD: ignore the normal columns, the nozzle stays normal
                               # to the z=0 plane (the mold is quasi-flat)
PIECE_VERTICAL_TOOL = False   # PIECE (plastic txt): False = orientation rebuilt from the
                               # normal (5-axis); True = vertical nozzle like the mold
VERTICAL_SIGN = -1             # (vertical mode) -1: TCP Z towards -Z of the part frame (nozzle
                               # down); +1: up
PIECE_TOOL_SIGN = -1          # piece: tool Z = PIECE_TOOL_SIGN * local surface normal.
                               # Successor of the old TOOL_AXIS/TOOL_SIGN pair: the export no
                               # longer carries tool axes, the nozzle axis IS the normal.
                               # -1: TCP Z points into the surface (nozzle against the part);
                               #    this matched the old defaults (TOOL_AXIS="w", TOOL_SIGN=-1
                               #    exporting w = -normal, so TCP Z = +normal OUT of the
                               #    surface... beware: the old combination was equivalent to
                               #    PIECE_TOOL_SIGN = +1 - flip here if the tool looks
                               #    reversed in the station)
                               # +1: TCP Z points out of the surface
FIXED_X = [0, 1, 0]            # reference vector (part frame): TCP X = its projection on the
                               # plane perpendicular to Z (constant rotation about the nozzle)
FIBER_TOOL_SIGN = -1                # fibers: tool Z = FIBER_TOOL_SIGN * normal

# --- Plastic ---
MIN_STEP = 0.05                     # mm: minimum distance between two kept points
DETOUR_Z_TOL = 0.2                  # mm: curve/connection point below print height - tol -> 'leave'
DETOUR_MAX_ZSPAN = 1.0              # mm: detour heuristic only valid on quasi-flat layers;
                                    # beyond this z spread over printed points the layer is
                                    # non-planar and detection is disabled
MAX_ANGLE = 5.0                     # deg: also keep a point if the orientation rotates more than this
LIFT_TRAVEL = True                  # True: approach/leave points below the print height are
                                    # raised to zref + Z_LIFT
Z_LIFT = 3.0                        # mm: clearance above the print height for travel moves

# --- Speeds / rounding ---
SPEED_PLASTIC = 20                  # mm/s on curve / connection types
SPEED_TRAVEL = 100                   # mm/s on approach / leave
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
TRACE_ONLY_PRINT = True             # RoboDK trace simulation only on curve/connection
                                    # (creates TraceOn/TraceOff programs in the station)
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
    normals: np.ndarray | None = None   # (N, 3) local surface normals exported by the slicer


@dataclass
class Move:
    pose: np.ndarray             # 4x4
    speed: float
    rounding: float
    label: str
    is_print: bool = False                   # True: material is being deposited (trace on)
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
def _load_path_txt(path):
    """Load a trajectory from the single whitespace-separated txt export:
    'x y z [nx ny nz] type [layer]' per row, segment type as a text label
    (see SlicedObject.export_path_to_txt). Returns (P, N, types, layers):
    normals N are (N, 3) or None when the file has no normal columns."""
    rows = []
    has_normals = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except (ValueError, IndexError):
                continue  # header / empty line
            has_n = len(parts) >= 7
            if has_normals is None:
                has_normals = has_n
            elif has_normals != has_n:
                raise SystemExit(f"{path}: inconsistent rows (normal columns)")
            if has_n:
                nx, ny, nz = float(parts[3]), float(parts[4]), float(parts[5])
                t = parts[6]
                layer = int(float(parts[7])) if len(parts) >= 8 else 0
                rows.append((x, y, z, nx, ny, nz, t, layer))
            else:
                t = parts[3]
                layer = int(float(parts[4])) if len(parts) >= 5 else 0
                rows.append((x, y, z, None, None, None, t, layer))
    if not rows:
        raise SystemExit(f"{path}: no usable rows found")
    P = np.array([r[0:3] for r in rows])
    N = np.array([r[3:6] for r in rows]) if has_normals else None
    types = np.array([r[6] for r in rows], dtype=object)
    layers = np.array([r[7] for r in rows])
    return P, N, types, layers


def _load_path_csv(path):
    """Load a trajectory from the single CSV export:
    'x,y,z,nx,ny,nz,type[,layer]' per row, segment type as a text label
    (see SlicedObject.export_path_to_csv). Returns (P, N, types, layers):
    normals N are (N, 3) or None when the file has no normal columns."""
    rows, _ = read_csv(path)
    if not rows:
        raise SystemExit(f"{path}: no usable rows found")
    ncols = len(rows[0])
    has_n = ncols >= 7
    P = np.array([[float(v) for v in r[0:3]] for r in rows])
    if has_n:
        N = np.array([[float(v) for v in r[3:6]] for r in rows])
        types = np.array([r[6] for r in rows], dtype=object)
        layers = np.array([int(float(r[7])) if ncols >= 8 and r[7] else 0 for r in rows])
    else:
        N = None
        types = np.array([r[3] for r in rows], dtype=object)
        layers = np.array([int(float(r[4])) if ncols >= 5 and r[4] else 0 for r in rows])
    return P, N, types, layers


def _load_path(path):
    """Dispatch on the file extension: .csv -> CSV export (plastic piece),
    anything else -> whitespace-separated txt export (mold)."""
    if str(path).lower().endswith(".csv"):
        return _load_path_csv(path)
    return _load_path_txt(path)


def load_mold(path, lift_travel=LIFT_TRAVEL):
    """Load the mold trajectory (normal-aware txt export, see
    SlicedMold.export_layers_to_txt). The normals are kept but ignored when
    VERTICAL_TOOL is set (the mold is quasi-flat)."""
    P, N, types, layers = _load_path_txt(path)

    # Detour detection per layer (a mold layer is quasi-flat by construction).
    ndet = relabel_detours(P, types, layers)
    if lift_travel:
        nlift = lift_travel(P, types, layers)
    else:
        nlift = 0

    # consecutive exact duplicates
    keep = (np.abs(np.diff(P, axis=0)).sum(1) > 1e-6) | (types[1:] != types[:-1])
    keep = np.r_[True, keep]
    P, types, layers = P[keep], types[keep], layers[keep]
    if N is not None:
        N = N[keep]
    path_obj = PlasticPath(points=P, normals=N, types=types, layers=layers,
                           n_detours=ndet, source=path)
    print(f"Mold: {len(P)} points from {os.path.basename(path)}, "
          f"layers {layers.min()}..{layers.max()}, {ndet} detour points -> leave, {nlift} lifted")
    return path_obj


# ----------------------------- plastic -----------------------------
def load_plastic(path, layers=PLASTIC_LAYERS, prefix="plastic ", lift_travel=LIFT_TRAVEL):
    """Load the plastic piece trajectory from the single CSV export
    ('x,y,z,nx,ny,nz,type,layer', see SlicedPiece.export_layers_to_csv)
    and keep the requested layers.
    layers: 'first', 'all', or a list of layer numbers."""
    P, N, T, layers_arr = _load_path(path)
    if N is None:
        raise SystemExit(f"{path}: expected the normal columns (x,y,z,nx,ny,nz,type,layer)")
    if len(P) == 0:
        raise SystemExit(f"No points in {path}")

    available = sorted(int(l) for l in np.unique(layers_arr))
    if layers == "first":
        wanted = [available[0]]
    elif layers == "all":
        wanted = available
    else:
        wanted = sorted(int(l) for l in layers)
    missing = [l for l in wanted if l not in available]
    if missing:
        raise SystemExit(f"Plastic layers {missing} not found (available: {available})")
    sel = np.isin(layers_arr, wanted)
    P, N, T, layers_arr = P[sel], N[sel], T[sel], layers_arr[sel]

    # Degenerate normals (zero length) -> copy from a valid neighbour.
    nbad = 0
    if N is not None:
        bad = np.linalg.norm(N, axis=1) < 1e-6
        if bad.any():
            good = ~bad
            for i in np.where(bad)[0]:
                j = good[:i][-1] if good[:i].any() else good[i:][0]
                N[i] = N[j]
            nbad = int(bad.sum())

    # Detours (below an already-laid line) -> 'leave': no printing.
    # Heuristic based on the dominant print height per layer: only valid
    # on a quasi-flat layer; on a non-planar layer (z spread over more
    # than DETOUR_MAX_ZSPAN) a lower z is legitimate, so detect nothing.
    is_print = np.isin(T, PRINT_TYPES)
    ndet = 0
    if is_print.any():
        for li in wanted:
            m = layers_arr == li
            if not (m & is_print).any():
                continue
            z_print = P[m & is_print][:, 2]
            if z_print.max() - z_print.min() <= DETOUR_MAX_ZSPAN:
                sel = np.where(m & is_print)[0]
                detour = np.zeros(len(P), bool)
                zref = _layer_zref(P[sel], T[sel], layers_arr[sel])
                for _, zr in zref.items():
                    detour |= m & is_print & (P[:, 2] < zr - DETOUR_Z_TOL)
                T[detour] = "leave"
                ndet += int((detour & m).sum())
                if lift_travel:
                    for _, zr in zref.items():
                        low = m & np.isin(T, ("approach", "leave")) & (P[:, 2] < zr - DETOUR_Z_TOL)
                        P[low, 2] = zr + Z_LIFT
            else:
                print(f"  non-planar layer {li} (z spread {z_print.max() - z_print.min():.1f} mm): "
                      "detour detection disabled")

    # consecutive exact duplicates
    keep = (np.abs(np.diff(P, axis=0)).sum(1) > 1e-6) | (T[1:] != T[:-1])
    keep = np.r_[True, keep]
    P, T = P[keep], T[keep]
    if N is not None:
        N = N[keep]

    path = PlasticPath(points=P, types=T, layers=layers_arr, n_detours=ndet,
                       source=path, normals=N)
    print(f"{prefix.capitalize()}: layers {wanted}, {len(P)} points, "
          f"{ndet} detour points -> leave, {nbad} invalid normals fixed")
    return path


def plastic_moves(p: PlasticPath, prefix="plastic "):
    """Build the TCP poses. Vertical mode (VERTICAL_TOOL=True): the nozzle
    stays normal to the z=0 plane of the part frame. Otherwise the pose is
    rebuilt from the local surface normal exported in the trajectory file
    ONLY: the nozzle axis is PIECE_TOOL_SIGN * normal. No travel tangent is
    used anywhere: the rotation about the nozzle axis is kept as constant
    as possible, which is what a non-cylindrical printhead needs to avoid
    spinning (and IK/simulation errors). Concretely: X = projection of a
    FIXED_X reference vector onto the plane perpendicular to Z; where Z
    is nearly parallel to FIXED_X, X is taken from the neighbouring pose
    (continuity) or from an axis-aligned fallback, so (X, Y, Z) stays a
    valid right-handed frame at every point."""
    P = p.points
    n = len(P)
    vertical = VERTICAL_TOOL if prefix.startswith("mold") else PIECE_VERTICAL_TOOL
    if vertical or p.normals is None:
        Z = np.tile([0.0, 0.0, float(VERTICAL_SIGN)], (n, 1))
        T = frames_from_z(P, Z, FIXED_X)
    else:
        Z = PIECE_TOOL_SIGN * p.normals
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        T = frames_from_z(P, Z, FIXED_X)
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
            is_print=printing,
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
        # 3) laydown (always printing while the fiber is being laid)
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
                label=f"{name} pt {i}", is_print=True, post=post))
        # 4) retreat after the end
        moves.append(Move(pose=hop(T[idx[-1]]), speed=SPEED_TRAVEL, rounding=0,
                          label=f"{name} end retreat"))
        print(f"  {name}: {len(P)} points -> {len(idx)} kept, length {L:.0f} mm"
              f"{'' if with_fibre else ' (no fiber)'}"
              f"{f', cut at point {cut_i}' if cut_i is not None else ''}")
    return moves, polylines


# ------------------------------- RoboDK -------------------------------
def ensure_trace_programs(RDK, robolink):
    """Create (if needed) the TraceOn / TraceOff Python programs in the station
    so the simulation only draws the deposited material (curve/connection)."""
    items = {}
    for name, state in (("TraceOn", "On"), ("TraceOff", "Off")):
        item = RDK.Item(name, robolink.ITEM_TYPE_PROGRAM_PYTHON)
        if not item.Valid():
            path = os.path.join(".", f"{name}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("from robodk import robolink\n"
                        "RDK = robolink.Robolink()\n"
                        f"RDK.Command('Trace', '{state}')\n")
            item = RDK.AddFile(path)
        items[name] = item
    return items


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
    # Open station folder: the data (piece_path.csv, mold_path.txt,
    # fibers_transformed/) lives next to the .rdk file.
    station_dir = RDK.getParam("PATH_OPENSTATION") or ""
    bases = [station_dir, os.getcwd()]
    piece_file = resolve_data_file(args.piece, bases)
    fibers_dir = resolve_data_dir(args.fibers, bases)
    print(f"Station folder: {station_dir or '?'}")
    print(f"Piece: {piece_file}\nFibers: {fibers_dir}")

    # --- mold (optional): printed first, plastic only ---
    mold_moves, mold_runs = [], []
    if args.print_mold:
        mold_file = resolve_data_file(args.mold, bases)
        if os.path.isfile(mold_file):
            mold = load_mold(mold_file, lift_travel=args.lift_travel)
            mold_moves, mold_runs = plastic_moves(mold, prefix="mold ")
            print(f"Mold: {len(mold_moves)} moves")
        else:
            print(f"Mold skipped: file {mold_file} not found")

    plastic = load_plastic(piece_file, plastic_layers(args), lift_travel=args.lift_travel)
    moves, plastic_runs = plastic_moves(plastic)

    # Orientation diagnostics: mean exported normal vs TCP Z of the first
    # piece move. They must be colinear (opposite by default, since TCP Z =
    # PIECE_TOOL_SIGN * normal with PIECE_TOOL_SIGN = -1). If the tool axis
    # looks wrong in the simulation, compare these two lines first: a sign
    # convention mismatch shows up as dot(z_tcp, mean_normal) ~= +1 instead
    # of -1 (fix: PIECE_TOOL_SIGN = +1, or reorient the station tool frame
    # so that its Z points along the nozzle, out of the tool).
    if plastic.normals is not None and len(moves):
        mean_n = plastic.normals.mean(axis=0)
        mean_n = mean_n / np.linalg.norm(mean_n)
        z_tcp = moves[0].pose[:3, 2]
        print(f"Orientation check: mean exported normal = {np.round(mean_n, 3).tolist()}")
        print(f"Orientation check: first piece TCP Z      = {np.round(z_tcp, 3).tolist()} "
              f"(dot with normal: {float(np.dot(z_tcp, mean_n)):+.2f}, "
              f"expected {-PIECE_TOOL_SIGN:+d})")

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

    if args.check_reach:
        tested = list(range(0, len(moves), CHECK_STEP))
        bad = [k for k in tested
               if len(robot.SolveIK(mat(moves[k].pose)).rows) < 6]
        print(f"IK test ({len(tested)} moves tested): {len(bad)} unreachable")
        for k in bad[:8]:
            p_abs = moves[k].pose[:3, 3] + np.array(frame.PoseAbs().Pos())
            print(f"  move {k + 1}: {moves[k].label} "
                  f"-> absolute approx {np.round(p_abs, 1).tolist()}")

    if args.draw_curves:
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

    if args.trace_only_print:
        ensure_trace_programs(RDK, robolink)
        RDK.Command("Trace", "Reset")      # clear previous traces
        trace_on = None

    cur_speed = cur_round = None
    for n, m in enumerate(moves):
        if m.speed != cur_speed:
            prog.setSpeed(m.speed)
            cur_speed = m.speed
        if m.rounding != cur_round:
            prog.setRounding(m.rounding)
            cur_round = m.rounding
        emit(m.pre)
        if args.trace_only_print:
            if m.is_print != trace_on:
                prog.RunInstruction("TraceOn" if m.is_print else "TraceOff",
                                    robolink.INSTRUCTION_CALL_PROGRAM)
                trace_on = m.is_print
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


def plastic_layers(args):
    """Resolve the plastic layer selection: 'first', 'all', a number, or a
    comma-separated list of numbers (e.g. '0,1,2')."""
    sel = args.plastic_layers
    if sel in ("first", "all"):
        return sel
    try:
        return [int(x) for x in str(sel).split(",")]
    except ValueError:
        return [int(sel)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--piece", default=PIECE_FILE,
                        help="piece CSV file (x,y,z,nx,ny,nz,type,layer)")
    parser.add_argument("--fibers", default=FIBERS_DIR,
                        help="folder holding layer_<i>/fiber_<i>_<j>.csv (fibers)")
    parser.add_argument("--plastic-layers", default=PLASTIC_LAYERS,
                        help="'first', 'all', a layer number, or '0,1,2'")
    parser.add_argument("--fiber-layer", default=FIBER_LAYER,
                        help="'first' or a fiber layer number")
    parser.add_argument("--mold", default=MOLD_FILE,
                        help="mold txt file (empty = skip the mold)")
    # Boolean options: each one is a module constant by default, flipped
    # from the command line with the paired store_true / store_false flags
    # (e.g. --no-mold sets print_mold=False, --mold-print restores True).
    bool_opts = (
        # (dest,       default,       '--off' flag,     '--on' flag,     help)
        ("print_mold", PRINT_MOLD, "--no-mold", "--print-mold",
         "print the mold first (piece printed on top of it)"),
        ("lift_travel", LIFT_TRAVEL, "--no-lift-travel", "--lift-travel",
         "raise approach/leave points below the print height (Z_LIFT clearance)"),
        ("check_reach", CHECK_REACH, "--no-check-reach", "--check-reach",
         "test IK reachability of the generated moves"),
        ("draw_curves", DRAW_CURVES, "--no-draw-curves", "--draw-curves",
         "draw the trajectories as RoboDK curves"),
        ("trace_only_print", TRACE_ONLY_PRINT, "--trace-all", "--trace-only-print",
         "RoboDK trace simulation only on curve/connection points"),
    )
    for dest, default, off, on, help_ in bool_opts:
        g = parser.add_mutually_exclusive_group()
        g.add_argument(off, dest=dest, action="store_false",
                       help=f"disable: {help_}")
        g.add_argument(on, dest=dest, action="store_true",
                       help=f"enable: {help_}")
        parser.set_defaults(**{dest: default})
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
