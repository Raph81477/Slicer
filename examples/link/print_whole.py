"""Whole-print visualization: mold, printing layers and fibers on one figure.

Combines the three visualizations into a single PyVista plotter:

- the mold: the two families of vertical slicing planes clipped to the
  shadow volume (toggleable),
- the printing layers of the part: offset layer surfaces and their
  printing paths (downsampled once, shared by the plot and the export),
- the fiber paths saved in the "fibers" directory (one color per layer).

All the configuration lives in the CONFIG block right below: change the
variables and booleans there to change the behaviour, there are no command
line arguments.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from oamc.constants import BANNER
from oamc.core import CompositeMaterial, CompositeModel
from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging

from slicer import SlicedMold, SlicedPiece
from slicer.utils.visualization import build_slicing_plane_patches

DIR = Path(__file__).parent.resolve()

# ======================================================================
# CONFIG — edit these variables and booleans to change the behaviour
# ======================================================================

# --- Model preparation ------------------------------------------------
MODEL_FILE = DIR / "link_smaller_mesh.dat"   # ANSYS .dat model to slice
ROTATE_AND_LIFT = True                       # align the layer normal with Z, then lift
N_LAYERS_UNDER_PIECE = 3                     # mold layers under the piece (lift)

# --- Printing parameters ----------------------------------------------
LAYER_HEIGHT = 0.35                          # layer thickness [mm]
LAYER_WIDTH = 0.70                           # bead width = nozzle diameter [mm]
N_LAYERS_CONTOUR = 5                         # contour passes (width = N x LAYER_WIDTH)
INFILL_ANGLE = 45.0                          # infill angle to the mean direction (0/45/90)
MEAN_DIRECTION_MODE = "dimensions"                # "fibers" or "dimensions"
APPROACH_DISTANCE = 1.0                       # approach/leave length [mm]
MAX_CONNECTION_LENGTH = 2.0                  # above this: leave + approach [mm]
SIMPLIFY_TOLERANCE = 0.5                     # RDP downsampling tolerance [mm] (0 disables)

# --- Edge detection (surface boundary used for the contours) ----------
FEATURE_ANGLE = 30.0                          # angle between adjacent faces [deg]
BOUNDARY_EDGES = True                         # include open borders
NON_MANIFOLD_EDGES = True                     # include non-manifold edges
VIS_EDGES = False                             # show the detected edges (no path computation)

# --- Mold -------------------------------------------------------------
GENERATE_MOLD = False                         # build the shadow volume + slicing planes
INFILL_PERCENTAGE = 5.0                      # mold infill [%]
PLANE_Z = 0.0                                # printing plane altitude
OFFSET_FACES = 0.05                          # ray origin offset [mm]

# --- Visualization ----------------------------------------------------
VIS_MOLD = GENERATE_MOLD                     # show the mold planes
VIS_LAYERS = None                            # piece layers to display (None = all, int = first N,
                                             # list = explicit indices e.g. [0, 2, 5])
VIS_LAYER_HEIGHT = 0.35                      # exaggerated spacing between plotted layers
VIS_FIBERS = False                            # show the fiber paths
VIS_MOLD_LAYERS = None                       # mold printing layers to display (None = none,
                                             # int = first N, list = explicit indices)

# ======================================================================

# --- Export ------------------------------------------------------------
EXPORT_FIBERS = False                         # save the transformed fibers to CSV
FIBERS_OUT = DIR / "fibers_transformed"      # output directory for the fibers
EXPORT_PLASTIC = True                        # save the plastic paths (per-layer CSVs)
PLASTIC_OUT = DIR / "trajectories"           # output directory for the plastic paths
EXPORT_PLASTIC_LAYERS = None                 # number of plastic layers to export (None = all)
EXPORT_MOLD = False                           # save the mold printing path to txt

# ======================================================================

PATH_COLORS = {
    "curve": "blue",
    "connection": "red",
    "approach": "green",
    "leave": "orange",
}

FIBER_COLORS = [
    "magenta",
    "purple",
    "darkcyan",
    "brown",
    "darkgreen",
    "navy",
    "olive",
    "deeppink",
]


def build_model() -> CompositeModel:
    """Load the .dat model, build the CompositeModel, rotate and lift it.

    The lift guarantees zmin > 0 then adds N_LAYERS_UNDER_PIECE layers
    below the piece, exactly as the GUI does."""
    parser = APDLParser(MODEL_FILE)

    model = parser.get_solid_model()[0]
    mold = parser.get_surface_mesh("MOLD")[0]

    matrix_material = IsotropicMaterial(
        E=2990,
        nu=0.39,
        rho=1.27e-9,
    )

    fiber_material = TransverselyIsotropicMaterial(
        E1=66550,
        E2=4900,
        nu12=0.32,
        G23=1680,
        G12=1880,
        rho=1.44e-9,
    )

    composite_material = CompositeMaterial(
        matrix_material=matrix_material,
        fiber_material=fiber_material,
    )

    model = CompositeModel(
        mesh=model.mesh,
        mold=mold,
        material=composite_material,
        dbc=model.dbc,
        nbc=model.nbc,
        fiber_diameter=LAYER_WIDTH,
        layer_height=LAYER_HEIGHT,
    )

    rot_mat = np.eye(3)
    lift_offset = 0.0

    if ROTATE_AND_LIFT or GENERATE_MOLD:
        rot_mat = SlicedPiece.get_rotation_matrix(model)
        SlicedMold.rotate_model(model, rot_mat)

        zmin = float(np.min(model.mesh.nodes[:, 2]))
        lift_offset = N_LAYERS_UNDER_PIECE * LAYER_HEIGHT - zmin
        SlicedMold.lift_model(model, offset=lift_offset)

    return model, rot_mat, lift_offset


def transform_fiber_paths(
    fiber_paths: dict[int, np.ndarray],
    rot_mat: np.ndarray,
    lift_offset: float = 0.0,
) -> dict[int, np.ndarray]:
    """Apply to raw (unrotated) fiber paths the same rigid transform as the
    model: rotation by rot_mat then translation along Z, matching
    SlicedMold.rotate_model and SlicedMold.lift_model."""
    transformed = {}
    for key, path in fiber_paths.items():
        points = (rot_mat @ path.T).T
        if lift_offset:
            points = points.copy()
            points[:, 2] += lift_offset
        transformed[key] = points
    return transformed


def slice_piece_paths(pieces: list[SlicedPiece]):
    """Build + downsample every layer's printing path, exactly as the GUI
    pipeline does. Returns piece_paths: one (segments, types) per layer."""
    contour_isovalues = [
        LAYER_WIDTH * (k + 0.5) for k in range(N_LAYERS_CONTOUR)
    ]

    piece_paths = []
    for i, piece in enumerate(pieces):
        piece.infill_angle_deg = INFILL_ANGLE
        piece.mean_direction_mode = MEAN_DIRECTION_MODE

        segments, _, types = piece.build_printing_path(
            i_surface=i,
            contour_width=N_LAYERS_CONTOUR * LAYER_WIDTH,
            approach_distance=APPROACH_DISTANCE,
            max_connection_length=MAX_CONNECTION_LENGTH,
            contour_isovalues=contour_isovalues,
            layer_width=LAYER_WIDTH,
        )

        # Downsample once: the plot and the exports below reuse these
        # already-simplified curves.
        segments, types = piece.analyze_and_downsample_path(
            segments,
            types,
            tolerance=SIMPLIFY_TOLERANCE or None,
            min_segment_length=2.0 * LAYER_WIDTH,
        )
        piece_paths.append((segments, types))

    return piece_paths


def resolve_layer_selection(selection, n_total: int, name: str) -> list[int]:
    """Resolve a layer selection into an explicit list of indices.

    - int: the first N layers
    - list/tuple/set of ints: exactly those indices (clipped to range)
    """
    if isinstance(selection, int):
        return list(range(min(selection, n_total)))
    if isinstance(selection, (list, tuple, set)):
        return sorted(i for i in selection if 0 <= i < n_total)
    raise TypeError(f"Invalid {name} layer selection: {selection!r}")


def add_edges(
    plotter: pv.Plotter,
    pieces: list[SlicedPiece],
    layer_indices: list[int],
) -> None:
    """Feature edges of the layer surfaces, as used for the contour
    computation. Purely geometric: no printing path is built, so this can
    be checked on its own before running the full path pipeline."""
    for i in layer_indices:
        edges = pieces[i].surface.extract_feature_edges(
            feature_angle=FEATURE_ANGLE,
            boundary_edges=BOUNDARY_EDGES,
            non_manifold_edges=NON_MANIFOLD_EDGES,
        )
        if edges.n_points == 0:
            continue
        plotter.add_mesh(
            edges,
            color="black",
            line_width=3,
            label="Detected edges" if i == layer_indices[0] else None,
        )


def add_printing_layers(
    plotter: pv.Plotter,
    pieces: list[SlicedPiece],
    piece_paths: list[tuple[list, list]],
    layer_indices: list[int],
) -> None:
    """Part layer surfaces and their (already downsampled) printing paths,
    for the selected layers only."""
    print(f"Total number of layers: {len(pieces)} (plotting {layer_indices})")

    normal = pieces[0].normal

    for i in layer_indices:
        piece = pieces[i]
        segments, types = piece_paths[i]

        surface_vis = piece.surface.copy()
        surface_vis.points = piece.surface.points + i * VIS_LAYER_HEIGHT * normal
        surface_vis.compute_normals(
            cell_normals=True, point_normals=False, inplace=True
        )

        plotter.add_mesh(
            surface_vis,
            color="lightblue",
            opacity=0.1,
            show_edges=True,
            label="Print surface",
        )

        print(f"Layer {i}: path = {len(segments)} segments")

        for segment, segment_type in zip(segments, types):
            plotter.add_mesh(
                pv.lines_from_points(segment, close=False),
                color=PATH_COLORS[segment_type],
                line_width=4 if segment_type == "connection" else 2,
            )


def load_transformed_fiber_layers(
    rot_mat: np.ndarray,
    lift_offset: float,
) -> dict[int, dict[int, np.ndarray]]:
    """Load every fiber layer from the fibers directory and apply the same
    rigid transform as the model (rotation then lift)."""
    fiber_root = DIR / "fibers"

    layer_dirs = sorted(
        (d for d in fiber_root.glob("layer_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )

    layers = {}
    for layer_dir in layer_dirs:
        layer = int(layer_dir.name.split("_")[-1])
        fiber_paths = SlicedPiece.load_fiber_paths(fiber_root.parent, layer=layer)
        layers[layer] = transform_fiber_paths(fiber_paths, rot_mat, lift_offset)
    return layers


def export_fibers(
    rot_mat: np.ndarray,
    lift_offset: float,
    out_directory: Path,
) -> list[Path]:
    """Export the (rotated + lifted) fiber paths to CSV files, in the same
    format as the oamc export (x, y, z, nx, ny, nz)."""
    layers = load_transformed_fiber_layers(rot_mat, lift_offset)

    written = []
    for layer, fiber_paths in layers.items():
        normals = {
            i: (rot_mat @ np.column_stack([
                np.zeros(len(path)),
                np.zeros(len(path)),
                -np.ones(len(path)),
            ]).T).T
            for i, path in fiber_paths.items()
        }
        written.extend(
            SlicedPiece.save_fiber_paths(
                fiber_paths,
                out_directory,
                layer=layer,
                normals=normals,
            )
        )

    print(f"Exported {len(written)} fiber files to {out_directory / 'fibers'}")
    return written


def export_plastic(
    pieces: list[SlicedPiece],
    piece_paths: list[tuple[list, list]],
    out_directory: Path,
) -> None:
    """Export the (already downsampled) plastic printing paths to CSV files,
    one subdirectory per layer (layer_i/segment_j.csv), in the same
    (already rotated and lifted) frame as the transformed fibers."""
    n_layers = EXPORT_PLASTIC_LAYERS or len(pieces)

    SlicedPiece.export_layers_to_csv(
        pieces[:n_layers],
        out_directory,
        piece_paths=piece_paths[:n_layers],
    )


def export_mold(mold_layers: list, out_directory: Path) -> None:
    """Export the (already sliced) mold printing path to txt."""
    SlicedMold.export_layers_to_txt(
        mold_layers,
        out_directory / "mold_path.txt",
        nozzle_diameter=LAYER_WIDTH,
        infill_percentage=INFILL_PERCENTAGE,
        approach_distance=APPROACH_DISTANCE,
        layer_height=LAYER_HEIGHT,
    )


def add_fibers(
    plotter: pv.Plotter,
    fiber_layers: dict[int, dict[int, np.ndarray]],
) -> None:
    """Fiber paths, one color per layer."""
    for i, (layer, fiber_paths) in enumerate(sorted(fiber_layers.items())):
        color = FIBER_COLORS[i % len(FIBER_COLORS)]

        for fiber_path in fiber_paths.values():
            plotter.add_mesh(
                pv.lines_from_points(fiber_path, close=False),
                color=color,
                line_width=2,
                label=f"Fibers layer {layer}" if fiber_paths else None,
            )


def main() -> None:
    enable_logging()
    print(BANNER)

    # 1. Model (rotation + lift, same convention as the GUI)
    model, rot_mat, lift_offset = build_model()

    # 2. Piece: slice (needed for the edges and the layer surfaces), then
    #    build + downsample the paths only when required.
    fiber_paths = SlicedPiece.load_fiber_paths(DIR, layer=1)
    fiber_paths = transform_fiber_paths(fiber_paths, rot_mat, lift_offset)
    pieces = SlicedPiece.slice_model(model, fiber_paths=fiber_paths)

    # Layer selections (piece + mold), resolved once
    piece_layer_indices = (
        resolve_layer_selection(VIS_LAYERS, len(pieces), "piece")
        if VIS_LAYERS is not None else []
    )

    need_paths = (
        bool(piece_layer_indices)
        or EXPORT_PLASTIC
    )
    piece_paths = slice_piece_paths(pieces) if need_paths else []

    # 3. Mold, computed ONCE: the shadow volume (for the planes and the
    #    layer slicing), the outer surface and the sliced layers are shared
    #    by the visualization and the export.
    shadow_volume = None
    mold_layers = []
    mold_layer_indices = []
    need_mold_layers = (VIS_MOLD_LAYERS is not None) or EXPORT_MOLD
    if GENERATE_MOLD:
        part_surface = SlicedMold._outer_surface(model)
        shadow_volume, _, _ = SlicedMold.get_shadow_volume_vertical(
            model,
            plane_z=PLANE_Z,
            offset_faces=OFFSET_FACES,
        )
        if need_mold_layers:
            mold_layers = SlicedMold.slice_shadow_volume(
                shadow_volume=shadow_volume,
                part_surface=part_surface,
                layer_height=LAYER_HEIGHT,
                nozzle_diameter=LAYER_WIDTH,
                infill_percentage=INFILL_PERCENTAGE,
                plane_z=PLANE_Z,
            )
            if VIS_MOLD_LAYERS is not None:
                mold_layer_indices = resolve_layer_selection(
                    VIS_MOLD_LAYERS, len(mold_layers), "mold"
                )

    # 4. Exports
    if EXPORT_FIBERS:
        export_fibers(rot_mat, lift_offset, FIBERS_OUT)

    if EXPORT_PLASTIC:
        export_plastic(pieces, piece_paths, PLASTIC_OUT)

    if mold_layers and EXPORT_MOLD:
        export_mold(mold_layers, PLASTIC_OUT)

    # 5. Visualization
    plotter = pv.Plotter(title="Whole print: mold + layers + fibers")

    if VIS_MOLD and shadow_volume is not None:
        plotter.add_mesh(
            shadow_volume.extract_surface(algorithm="dataset_surface"),
            color="gray",
            opacity=0.1,
            show_edges=False,
            label="Shadow volume",
        )

        patches_x, patches_y = build_slicing_plane_patches(
            shadow_volume,
            nozzle_diameter=LAYER_WIDTH,
            infill_percentage=INFILL_PERCENTAGE,
        )
        for patch in patches_x:
            plotter.add_mesh(
                patch, color="blue", opacity=0.35, show_edges=False,
                label="Planes normal to X",
            )
        for patch in patches_y:
            plotter.add_mesh(
                patch, color="red", opacity=0.35, show_edges=False,
                label="Planes normal to Y",
            )

    if VIS_EDGES:
        edge_indices = piece_layer_indices or list(range(len(pieces)))[:1]
        add_edges(plotter, pieces, edge_indices)

    if piece_layer_indices:
        add_printing_layers(plotter, pieces, piece_paths, piece_layer_indices)

    # Selected mold printing layers. Visualizing several at once can
    # crash PyVista, keep the selection small.
    if mold_layer_indices:
        for j in mold_layer_indices:
            layer = mold_layers[j]
            segments, types = layer.build_support_path(
                nozzle_diameter=LAYER_WIDTH,
                infill_percentage=INFILL_PERCENTAGE,
                approach_distance=APPROACH_DISTANCE,
                layer_height=LAYER_HEIGHT,
            )
            first = j == mold_layer_indices[0]
            for seg, seg_type in zip(segments, types):
                plotter.add_mesh(
                    pv.lines_from_points(seg, close=False),
                    color=PATH_COLORS[seg_type],
                    line_width=4 if seg_type == "connection" else 2,
                    label=f"Mold path (layer q={layer.q})" if first else None,
                )

    if VIS_FIBERS:
        fiber_layers = load_transformed_fiber_layers(rot_mat, lift_offset)
        add_fibers(plotter, fiber_layers)

    plotter.add_legend()
    plotter.show()


if __name__ == "__main__":
    main()
