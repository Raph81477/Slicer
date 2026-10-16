"""Génère un programme RoboDK : 1re couche de plastique, puis 1re couche de fibres de carbone par-dessus.

Usage : ouvrir RobotPrint_Printhead.rdk dans RoboDK, puis lancer ce script (RoboDK doit tourner).
Unités : mm. Le code KRL (anisoprint_*) n'apparaît que dans le .src exporté, pas en simulation.

Fichiers attendus (trajectories/, à côté de ce script) :
  trajectories/layer_<i>/segment_<j>.csv   x,y,z,type,layer   (plastique, header sur la 1re ligne)
  fibers/layer_<i>/fiber_<i>_<j>.csv        x,y,z,nx,ny,nz    (fibres, 5 décimales)
Plastique et fibres doivent être dans le même repère (repère pièce).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

# ------------------------- PARAMÈTRES -------------------------
TRAJECTORIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
FIBERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fibers")

PLASTIC_LAYER = "first"        # "first" = première couche trouvée, ou un n° de couche
FIBER_LAYER = "first"          # "first" = plus petit i dans fibers/layer_i, ou un n° de couche

FRAME_NAME = "PieceFrame"
FRAME_XYZRPY = [800, 0, 0, 0, 0, 0]   # ABSOLU dans la station -> À AJUSTER
FORCE_FRAME_POSE = True

# --- Orientation ---
VERTICAL_SIGN = -1             # plastique : Z du TCP vers -Z pièce (buse vers le bas)
FIBER_TOOL_SIGN = -1           # fibres : Z du TCP = FIBER_TOOL_SIGN * normale (-1 si la normale sort de la pièce)
FIXED_X = [0, 1, 0]            # vecteur de référence pour X du TCP (évite les rotations inutiles autour de la buse)

# --- Plastique ---
MIN_STEP = 3.0                 # mm : distance mini entre deux points gardés
DETOUR_Z_TOL = 0.2             # mm : point curve/connection plus bas que la hauteur d'impression - tol = contournement -> 'leave'

# --- Vitesses / arrondi ---
SPEED_PLASTIC = 50             # mm/s (0,05 m/s)
SPEED_TRAVEL = 80              # mm/s (0,08 m/s)
ROUNDING = 0.5                 # mm, seulement entre deux points d'impression consécutifs

# --- Fibres ---
FIBER_MIN_STEP = 2.0           # mm
FIBER_MAX_ANGLE = 5.0          # deg
FIBER_ROUNDING = 0.5           # mm (0 sur le point de coupe et le dernier point)
SPEED_FIBER = 5                # mm/s (0,005 m/s) fibre posée
SPEED_FIBER_NO_FIBRE = 20      # mm/s (0,02 m/s) tracé trop court : plastique composite sans fibre
MINIMAL_COMPOSITE_LENGTH = 80  # mm : en dessous, pas de fibre (comme le postprocessor)
DISTANCE_CUT_TO_END = 50       # mm : la fibre est coupée à cette distance avant la fin du tracé
FIBER_HOP = 5                  # mm : dégagement le long de la buse entre deux fibres
LIFT_Z = 10                    # mm : montée verticale après le plastique, avant le changement d'extrudeur

# --- Code KRL (identique au postprocessor) ---
PLASTIC_TEMP, COMP_TEMP = 230, 250
INIT_CODE = ["feeders_init(#Anisoprint_Plastic)",
             f"anisoprint_temp(TRUE, {PLASTIC_TEMP}, FALSE, {COMP_TEMP})"]
PLASTIC_ON = "anisoprint_set_feed(13,0,0)"
EXTRUDE_OFF = "anisoprint_set_feed(0,0,0)"
CHANGE_TO_COMPOSITE = f"aniso_change_extruder(FALSE, {PLASTIC_TEMP}, {COMP_TEMP})"
FIBER_ON = "anisoprint_set_feed(0,100,50)"
FIBER_ON_NO_FIBRE = "anisoprint_set_feed(0,0,50)"
CUT_CODE = ["WAIT SEC 0.3", "anisoprint_retract(0,3,0,100)", "anisoprint_cut()", "WAIT SEC 0.7"]
PLASTIC_OFF_DELAY = 500        # ms après arrêt plastique
FIBER_OFF_DELAY = "WAIT SEC 0.5"

CHECK_REACH = True
CHECK_STEP = 2
DRAW_CURVES = True             # plastique en rouge, fibres en bleu (courbes RoboDK)
# --------------------------------------------------------------

PRINT_TYPES = ("curve", "connection")


# ----------------------------- données -----------------------------
@dataclass
class PlasticPath:
    points: np.ndarray          # (N, 3)
    types: np.ndarray          # (N,) chaînes : curve / connection / approach / leave
    layer: int
    n_detours: int = 0
    source: str = ""


@dataclass
class Move:
    pose: np.ndarray           # 4x4
    speed: float
    rounding: float
    label: str
    pre: list = field(default_factory=list)     # ("comment"|"code"|"pause", valeur)
    post: list = field(default_factory=list)


def read_csv(path):
    """Lit un CSV avec en-tête optionnel -> (rows, header|None).

    Renvoie les lignes brutes en chaînes : le type peut être un libellé
    ("curve", ...) ou une valeur numérique, selon l'export."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    try:
        float(rows[0][0])
        return rows, None
    except ValueError:
        return rows[1:], rows[0]


# ----------------------- utilitaires géométrie -----------------------
def frames_from_z(P, Z, ref):
    """Matrices 4x4 : Z donné, X = projection de ref dans le plan normal à Z."""
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    ref = np.tile(np.array(ref, float), (len(Z), 1))
    X = ref - (ref * Z).sum(1, keepdims=True) * Z
    bad = np.linalg.norm(X, axis=1) < 0.1          # Z presque // ref : référence de secours
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
    """Indices gardés : début, fin, points forcés, et tout point à > step ou > max_angle du dernier gardé."""
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


# ----------------------------- plastique -----------------------------
def load_plastic(traj_dir=TRAJECTORIES_DIR, layer=PLASTIC_LAYER):
    """Charge tous les segment_<j>.csv de la couche demandée en une trajectoire continue."""
    layer_dirs = {}
    for p in glob.glob(os.path.join(traj_dir, "layer_*")):
        m = re.fullmatch(r"layer_(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            layer_dirs[int(m.group(1))] = p
    if not layer_dirs:
        raise SystemExit(f"Aucun dossier layer_<i> dans {traj_dir}")
    li = min(layer_dirs) if layer == "first" else int(layer)
    if li not in layer_dirs:
        raise SystemExit(f"Couche plastique {li} introuvable (dispo : {sorted(layer_dirs)})")

    files = []
    for p in glob.glob(os.path.join(layer_dirs[li], "segment_*.csv")):
        m = re.fullmatch(r"segment_(\d+)\.csv", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if not files:
        raise SystemExit(f"Aucun segment_<j>.csv dans {layer_dirs[li]}")

    points, types = [], []
    for _, path in files:
        rows, header = read_csv(path)
        if not rows or len(rows[0]) < 4:
            raise SystemExit(f"{path} : attendu x,y,z,type (au moins 4 colonnes)")
        pts = np.array([[float(v) for v in r[0:3]] for r in rows])
        # Colonne 'type' : libellé texte (export "labels") ou valeur
        # numérique (export "orientation", 0 = leave, 1 = curve).
        if header is not None and "type" in header:
            raw = [r[header.index("type")] for r in rows]
        else:
            raw = [r[3] for r in rows]
        try:
            float(raw[0])
            labels = ["curve" if float(v) > 0.5 else "leave" for v in raw]
        except ValueError:
            labels = raw
        points.append(pts)
        types.append(np.array(labels, dtype=object))
    P = np.vstack(points)
    T = np.concatenate(types)
    if len(P) == 0:
        raise SystemExit(f"Aucun point dans {layer_dirs[li]}")

    # contournements (sous une ligne déjà posée) -> 'leave' : pas d'impression
    is_print = np.isin(T, PRINT_TYPES)
    if is_print.any():
        z = np.round(P[is_print][:, 2], 3)
        zref = float(np.bincount(np.round((z - z.min()) * 1000).astype(int)).argmax()) / 1000 + z.min()
        detour = is_print & (P[:, 2] < zref - DETOUR_Z_TOL)
        T[detour] = "leave"
    else:
        detour = np.zeros(len(P), bool)

    # doublons exacts consécutifs
    keep = (np.abs(np.diff(P, axis=0)).sum(1) > 1e-6) | (T[1:] != T[:-1])
    keep = np.r_[True, keep]
    P, T = P[keep], T[keep]

    path = PlasticPath(points=P, types=T, layer=li, n_detours=int(detour.sum()),
                       source=layer_dirs[li])
    print(f"Plastique : couche {li}, {len(files)} segments CSV, {len(P)} points, "
          f"{path.n_detours} points de contournement -> leave")
    return path


def plastic_moves(p: PlasticPath):
    Z = np.tile([0.0, 0.0, float(VERTICAL_SIGN)], (len(p.points), 1))
    T = frames_from_z(p.points, Z, FIXED_X)
    types = p.types

    # décimation : on garde aussi les changements de type
    keep, last = [0], 0
    for i in range(1, len(T)):
        if types[i] != types[i - 1]:
            if keep[-1] != i - 1:
                keep.append(i - 1)
            keep.append(i)
            last = i
        elif (np.linalg.norm(T[i, :3, 3] - T[last, :3, 3]) >= MIN_STEP):
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
            pre.append(("comment", f"plastique couche {p.layer} - {t}"))
            prev_type = t
        if printing and not printing_prev:
            pre.append(("code", PLASTIC_ON))
        if not printing and printing_prev:
            pre.append(("code", EXTRUDE_OFF))
            pre.append(("pause", PLASTIC_OFF_DELAY))
        printing_prev = printing
        moves.append(Move(
            pose=T[i],
            speed=SPEED_PLASTIC if t not in ("approach", "leave") else SPEED_TRAVEL,
            rounding=ROUNDING if (t == "curve" and nxt in PRINT_TYPES) else 0,
            label=f"plastique {t} x={p.points[i,0]:.1f} y={p.points[i,1]:.1f} z={p.points[i,2]:.1f}",
            pre=pre, post=[]))
    if printing_prev:
        moves[-1].post.append(("code", EXTRUDE_OFF))

    # polylignes d'impression pour le tracé
    runs, cur = [], []
    is_p = np.isin(types, PRINT_TYPES)
    for i in range(len(p.points)):
        if is_p[i]:
            cur.append(p.points[i])
        else:
            if len(cur) > 1:
                runs.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        runs.append(np.array(cur))
    print(f"Plastique : {len(p.points)} points -> {len(moves)} mouvements")
    return moves, runs


# ------------------------------- fibres -------------------------------
def load_fiber(path):
    rows, _ = read_csv(path)
    try:
        A = np.array([[float(v) for v in r] for r in rows], ndmin=2)
    except ValueError as e:
        raise SystemExit(f"{path} : {e}")
    if A.ndim != 2 or A.shape[1] < 6:
        raise SystemExit(f"{path} : attendu x,y,z,nx,ny,nz (6 colonnes)")
    P, N = A[:, 0:3], A[:, 3:6]
    keep = np.r_[True, np.abs(np.diff(P, axis=0)).sum(1) > 1e-6]   # doublons consécutifs
    return P[keep], N[keep]


def find_fiber_files(fiber_dir=FIBERS_DIR, layer=FIBER_LAYER):
    layers = {}
    for p in glob.glob(os.path.join(fiber_dir, "layer_*")):
        m = re.fullmatch(r"layer_(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            layers[int(m.group(1))] = p
    if not layers:
        raise SystemExit(f"Aucun dossier layer_<i> dans {fiber_dir}")
    li = min(layers) if layer == "first" else int(layer)
    if li not in layers:
        raise SystemExit(f"Couche de fibres {li} introuvable (dispo : {sorted(layers)})")
    files = []
    for p in glob.glob(os.path.join(layers[li], f"fiber_{li}_*.csv")):
        m = re.fullmatch(rf"fiber_{li}_(\d+)\.csv", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if not files:
        raise SystemExit(f"Aucun fichier fiber_{li}_<j>.csv dans {layers[li]}")
    print(f"Fibres : couche {li}, {len(files)} fibres")
    return li, [p for _, p in files]


def fiber_moves(files):
    moves, polylines = [], []
    for path in files:
        P, N = load_fiber(path)
        if len(P) < 2:
            print("  fibre ignorée (moins de 2 points) :", os.path.basename(path))
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
            H[:3, 3] = Tk[:3, 3] - FIBER_HOP * Tk[:3, 2]       # on recule le long de -Z du TCP
            return H

        # 1) dégagement au-dessus du début
        moves.append(Move(
            pose=hop(T[0]), speed=SPEED_TRAVEL, rounding=0,
            pre=[("comment", f"{name} - {L:.0f} mm, {'fibre' if with_fibre else 'SANS fibre (trop court)'}")],
            label=f"{name} dégagement début"))
        # 2) début de la fibre ; l'extrusion démarre une fois arrivé
        speed = SPEED_FIBER if with_fibre else SPEED_FIBER_NO_FIBRE
        moves.append(Move(
            pose=T[idx[0]], speed=SPEED_TRAVEL, rounding=0,
            post=[("code", FIBER_ON if with_fibre else FIBER_ON_NO_FIBRE)],
            label=f"{name} début"))
        # 3) tracé
        for k, i in enumerate(idx[1:], start=1):
            last = (k == len(idx) - 1)
            is_cut = (i == cut_i)
            post = []
            if is_cut:
                post += [("code", c) for c in CUT_CODE]
            if last:
                post += [("code", EXTRUDE_OFF), ("code", FIBER_OFF_DELAY)]
            moves.append(Move(
                pose=T[i], speed=speed,
                rounding=0 if (last or is_cut) else FIBER_ROUNDING,
                label=f"{name} pt {i}", post=post))
        # 4) dégagement après la fin
        moves.append(Move(pose=hop(T[idx[-1]]), speed=SPEED_TRAVEL, rounding=0,
                          label=f"{name} dégagement fin"))
        print(f"  {name} : {len(P)} points -> {len(idx)} gardés, longueur {L:.0f} mm"
              f"{'' if with_fibre else ' (sans fibre)'}"
              f"{f', coupe au point {cut_i}' if cut_i is not None else ''}")
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
    plastic = load_plastic(args.trajectories, args.plastic_layer)
    moves, plastic_runs = plastic_moves(plastic)

    _, files = find_fiber_files(args.fibers, args.fiber_layer)
    fmoves, fiber_lines = fiber_moves(files)

    # montée verticale après le plastique, puis changement d'extrudeur
    lift = moves[-1].pose.copy()
    lift[2, 3] += LIFT_Z
    moves.append(Move(
        pose=lift, speed=SPEED_TRAVEL, rounding=0,
        pre=[("comment", "changement d'extrudeur : plastique -> composite")],
        post=[("code", CHANGE_TO_COMPOSITE)], label="montée avant fibres"))
    moves += fmoves

    RDK = robolink.Robolink()
    robot = RDK.ItemUserPick("Choisir le robot", robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise SystemExit("Aucun robot choisi")
    tool = robot.getLink(robolink.ITEM_TYPE_TOOL)
    if not tool.Valid():
        raise SystemExit("Aucun outil attaché au robot : attache la tête d'impression dans RoboDK")

    frame = RDK.Item(FRAME_NAME, robolink.ITEM_TYPE_FRAME)
    if not frame.Valid():
        frame = RDK.AddFrame(FRAME_NAME)
        force = True
    else:
        force = FORCE_FRAME_POSE
    if force:
        frame.setParent(RDK.ActiveStation())
        frame.setPoseAbs(robomath.xyzrpw_2_pose(FRAME_XYZRPY))
    print("Repère pièce (absolu) :", np.round(frame.PoseAbs().Pos(), 1),
          "| base du robot (absolu) :", np.round(robot.PoseAbs().Pos(), 1))

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
        print(f"Test IK ({len(tested)} mouvements testés) : {len(bad)} inaccessibles")
        for k in bad[:8]:
            print(f"  mouvement {k + 1} : {moves[k].label}")

    if DRAW_CURVES:
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
            emit([("code", c) for c in INIT_CODE])      # comme le postprocessor : après le premier PTP
        else:
            prog.MoveL(mat(m.pose))
        emit(m.post)

    try:
        print("Vérification RoboDK :", prog.Update()[-1])
    except Exception as e:
        print("prog.Update() indisponible :", e)
    print(f"Programme 'PrintPlasticFiber' créé : {len(moves)} mouvements. Lance-le ou simule-le.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", default=TRAJECTORIES_DIR,
                        help="dossier contenant layer_<i>/segment_<j>.csv (plastique)")
    parser.add_argument("--fibers", default=FIBERS_DIR,
                        help="dossier contenant layer_<i>/fiber_<i>_<j>.csv (fibres)")
    parser.add_argument("--plastic-layer", default=PLASTIC_LAYER,
                        help="'first' ou numéro de couche plastique")
    parser.add_argument("--fiber-layer", default=FIBER_LAYER,
                        help="'first' ou numéro de couche de fibres")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
