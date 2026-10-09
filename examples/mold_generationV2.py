"""Lightweight visualization of the mold slicing planes.

Instead of building (and plotting) every printing layer like
mold_generation.py does, this example only displays the mold's slicing
planes: the two families of vertical planes (normal to X and normal to
Y) used by the infill, clipped to the shadow volume. Only the part of
each plane lying inside the shadow volume is drawn, as a filled patch.
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

DIR = Path(__file__).parent.resolve()

NOZZLE_DIAMETER = 0.70
INFILL_PERCENTAGE = 5.0
PLANE_Z = 0.0
OFFSET_FACES = 0.05


def _fill_loop(points: np.ndarray, tolerance: float = 1e-6) -> pv.PolyData | None:
    """Triangulate a plane contour into a filled patch, if it is a closed loop."""
    if len(points) < 3:
        return None
    if np.linalg.norm(points[0] - points[-1]) > tolerance * max(
        1.0, np.ptp(points, axis=0).max()
    ):
        return None
    return pv.PolyData(points).delaunay_2d()


def build_slicing_plane_patches(
    shadow_volume: pv.UnstructuredGrid,
    nozzle_diameter: float = NOZZLE_DIAMETER,
    infill_percentage: float = INFILL_PERCENTAGE,
    fill: bool = True,
) -> tuple[list[pv.PolyData], list[pv.PolyData]]:
    """Intersect the shadow volume with the two families of slicing planes.

    The plane spacing is the same as the one used by the infill, and the
    intersection with all the planes of one family is computed in a
    single contour pass (SlicedMold._cut_axis_planes).

    Returns
    -------
    patches_x : filled patches of the planes normal to X
    patches_y : filled patches of the planes normal to Y
    """
    surface = shadow_volume.extract_surface(algorithm="dataset_surface")
    surface = surface.triangulate().clean(tolerance=1e-6)

    spacing = SlicedMold.infill_spacing(nozzle_diameter, infill_percentage)
    bounds = surface.bounds
    levels_x = np.arange(bounds[0], bounds[1] + spacing, spacing)
    levels_y = np.arange(bounds[2], bounds[3] + spacing, spacing)
    print(f"Slicing planes: {len(levels_x)} normal to X, "
          f"{len(levels_y)} normal to Y (spacing {spacing:.3f} mm)")

    patches_x = []
    patches_y = []

    for axis, levels, patches in (
        (0, levels_x, patches_x),
        (1, levels_y, patches_y),
    ):
        for poly in SlicedMold._cut_axis_planes(surface, axis, levels):
            patch = _fill_loop(poly) if fill else None
            if patch is not None:
                patches.append(patch)
            elif len(poly) >= 2:
                patches.append(pv.lines_from_points(poly, close=False))

    print(f"Patches: {len(patches_x)} (X), {len(patches_y)} (Y)")
    return patches_x, patches_y


def main() -> None:
    enable_logging()

    print(BANNER)

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
        layer_height=1.0,
    )

    SlicedMold.rotate_model(model, SlicedPiece.get_rotation_matrix(model))
    SlicedMold.lift_model(model, offset=65)

    shadow_volume, _, _ = SlicedMold.get_shadow_volume_vertical(
        model,
        plane_z=PLANE_Z,
        offset_faces=OFFSET_FACES,
    )

    patches_x, patches_y = build_slicing_plane_patches(
        shadow_volume,
        nozzle_diameter=NOZZLE_DIAMETER,
        infill_percentage=INFILL_PERCENTAGE,
    )

    plotter = pv.Plotter(title="Mold slicing planes (shadow volume only)")

    shadow_surface = shadow_volume.extract_surface(algorithm="dataset_surface")
    plotter.add_mesh(
        shadow_surface,
        color="gray",
        opacity=0.1,
        show_edges=False,
        label="Shadow volume",
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

    plotter.add_legend()
    plotter.show()


if __name__ == "__main__":
    main()
