"""Whole-print visualization: mold, printing layers and fibers on one figure.

Combines the three existing visualizations into a single PyVista plotter:

- the mold, displayed as in mold_generationV2.py: the two families of
  vertical slicing planes clipped to the shadow volume (toggleable),
- the printing layers of the part, displayed as in print_layers.py:
  offset layer surfaces and their printing paths,
- the fiber paths saved in the "fibers" directory (one color per layer).
"""

import argparse
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

NOZZLE_DIAMETER = 0.70
INFILL_PERCENTAGE = 5.0
PLANE_Z = 0.0
OFFSET_FACES = 0.05
LIFT_OFFSET = 20.0

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
    parser = APDLParser(DIR / "c_beam_large_thin.dat")

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
        fiber_diameter=NOZZLE_DIAMETER,
        layer_height=0.35,
    )

    rot_mat = SlicedPiece.get_rotation_matrix(model)
    SlicedMold.rotate_model(model, rot_mat)
    SlicedMold.lift_model(model, offset=LIFT_OFFSET)
    return model, rot_mat


def add_mold(plotter: pv.Plotter, model: CompositeModel) -> None:
    """Shadow volume and its two families of slicing planes, as in
    mold_generationV2.py."""
    shadow_volume, _, _ = SlicedMold.get_shadow_volume_vertical(
        model,
        plane_z=PLANE_Z,
        offset_faces=OFFSET_FACES,
    )

    plotter.add_mesh(
        shadow_volume.extract_surface(algorithm="dataset_surface"),
        color="gray",
        opacity=0.1,
        show_edges=False,
        label="Shadow volume",
    )

    patches_x, patches_y = build_slicing_plane_patches(
        shadow_volume,
        nozzle_diameter=NOZZLE_DIAMETER,
        infill_percentage=INFILL_PERCENTAGE,
    )

    for patch in patches_x:
        plotter.add_mesh(
            patch,
            color="blue",
            opacity=0.35,
            show_edges=False,
            label="Planes normal to X",
        )

    for patch in patches_y:
        plotter.add_mesh(
            patch,
            color="red",
            opacity=0.35,
            show_edges=False,
            label="Planes normal to Y",
        )


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


def add_printing_layers(
    plotter: pv.Plotter,
    model: CompositeModel,
    rot_mat: np.ndarray,
    n_layers: int,
    vis_layer_height: float = 0.35,
) -> None:
    """Part layer surfaces and printing paths, as in print_layers.py."""
    fiber_paths = SlicedPiece.load_fiber_paths(DIR, layer=1)
    fiber_paths = transform_fiber_paths(
        fiber_paths, rot_mat, lift_offset=LIFT_OFFSET
    )
    pieces = SlicedPiece.slice_model(model, fiber_paths=fiber_paths)

    n_layers = min(n_layers, len(pieces))
    print(f"Total number of layers: {len(pieces)} (plotting {n_layers})")

    normal = pieces[0].normal

    for i, piece in enumerate(pieces[:n_layers]):
        surface_vis = piece.surface.copy()
        surface_vis.points = piece.surface.points + i * vis_layer_height * normal
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

        path_segments, _, segment_types = piece.build_printing_path(
            i_surface=i,
            contour_width=3.0,
            approach_distance=1.0,
            max_connection_length=4 * 0.5,
        )

        print(f"Layer {i}: full path = {len(path_segments)} segments")

        for segment, segment_type in zip(path_segments, segment_types):
            plotter.add_mesh(
                pv.lines_from_points(segment, close=False),
                color=PATH_COLORS[segment_type],
                line_width=4 if segment_type == "connection" else 2,
            )


def load_transformed_fiber_paths(
    rot_mat: np.ndarray,
    fiber_root: Path | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """Load every fiber layer from the fibers directory and apply the same
    rigid transform as the model (rotation then lift)."""
    fiber_root = fiber_root or DIR / "fibers"

    layer_dirs = sorted(
        (d for d in fiber_root.glob("layer_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )

    layers = {}
    for layer_dir in layer_dirs:
        layer = int(layer_dir.name.split("_")[-1])
        fiber_paths = SlicedPiece.load_fiber_paths(fiber_root.parent, layer=layer)
        layers[layer] = transform_fiber_paths(
            fiber_paths, rot_mat, lift_offset=LIFT_OFFSET
        )
    return layers


def export_fibers(
    rot_mat: np.ndarray,
    out_directory: Path,
    fiber_root: Path | None = None,
) -> list[Path]:
    """Export the (rotated + lifted) fiber paths to CSV files, in the same
    format as the oamc export (x, y, z, nx, ny, nz)."""
    layers = load_transformed_fiber_paths(rot_mat, fiber_root)

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
    model: CompositeModel,
    out_file: Path,
    n_layers: int | None = None,
    simplify_tolerance: float | None = None,
) -> None:
    """Export the plastic printing paths of every layer to a single
    trajectory file (x y z nx ny nz type layer per row, see
    SlicedPiece.export_layers_to_txt), in the same (already rotated and
    lifted) frame as the transformed fibers. The model itself has already
    been rotated and lifted in place by build_model."""
    fiber_paths = SlicedPiece.load_fiber_paths(DIR, layer=1)
    pieces = SlicedPiece.slice_model(model, fiber_paths=fiber_paths)

    if n_layers is not None:
        pieces = pieces[:n_layers]

    SlicedPiece.export_layers_to_txt(
        pieces,
        out_file,
        contour_width=3.0,
        approach_distance=1.0,
        max_connection_length=4 * 0.5,
        include_normal=True,
        simplify_tolerance=simplify_tolerance,
    )


def add_fibers(
    plotter: pv.Plotter,
    rot_mat: np.ndarray,
    fiber_root: Path | None = None,
) -> None:
    """Fiber paths saved in the fibers directory, one color per layer."""
    fiber_root = fiber_root or DIR / "fibers"

    layer_dirs = sorted(
        (d for d in fiber_root.glob("layer_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )

    for i, layer_dir in enumerate(layer_dirs):
        layer = int(layer_dir.name.split("_")[-1])
        color = FIBER_COLORS[i % len(FIBER_COLORS)]

        fiber_paths = SlicedPiece.load_fiber_paths(fiber_root.parent, layer=layer)
        fiber_paths = transform_fiber_paths(
            fiber_paths, rot_mat, lift_offset=LIFT_OFFSET
        )

        for fiber_path in fiber_paths.values():
            plotter.add_mesh(
                pv.lines_from_points(fiber_path, close=False),
                color=color,
                line_width=2,
                label=f"Fibers layer {layer}" if fiber_paths else None,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display mold, printing layers and fibers on one figure."
    )
    parser.add_argument(
        "--no-mold",
        action="store_true",
        help="Do not display the mold (shadow volume and slicing planes).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of part printing layers to display (default: 2).",
    )
    parser.add_argument(
        "--no-export-fibers",
        action="store_true",
        help=(
            "Skip the export of the rotated and lifted fiber paths to "
            "CSV files (exported by default)."
        ),
    )
    parser.add_argument(
        "--fibers-out",
        type=Path,
        default=DIR / "fibers_transformed",
        help=(
            "Output directory for --export-fibers "
            f"(default: {DIR / 'fibers_transformed'})."
        ),
    )
    parser.add_argument(
        "--no-export-plastic",
        action="store_true",
        help=(
            "Skip the export of the plastic printing paths to "
            "the trajectory file (exported by default)."
        ),
    )
    parser.add_argument(
        "--plastic-out",
        type=Path,
        default=DIR / "piece_path.txt",
        help=(
            "Output file for the plastic trajectory, "
            "'x y z nx ny nz type layer' per row "
            f"(default: {DIR / 'piece_path.txt'})."
        ),
    )
    parser.add_argument(
        "--export-plastic-layers",
        type=int,
        default=None,
        help=(
            "Number of plastic layers to export (default: all)."
        ),
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.5,
        help=(
            "Ramer-Douglas-Peucker tolerance (mm) applied to the plastic "
            "paths before export (default: 0.5, 0 disables)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    enable_logging()
    print(BANNER)

    args = parse_args()

    model, rot_mat = build_model()

    if not args.no_export_fibers:
        export_fibers(rot_mat, args.fibers_out)

    if not args.no_export_plastic:
        export_plastic(
            model,
            args.plastic_out,
            n_layers=args.export_plastic_layers,
            simplify_tolerance=args.simplify_tolerance or None,
        )

    plotter = pv.Plotter(title="Whole print: mold + layers + fibers")

    if not args.no_mold:
        add_mold(plotter, model)

    add_printing_layers(plotter, model, rot_mat, n_layers=args.layers)
    add_fibers(plotter, rot_mat)

    plotter.add_legend()
    plotter.show()


if __name__ == "__main__":
    main()
