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


def _plane_coords(points: np.ndarray) -> tuple[int, np.ndarray]:
    """2D coordinates of a plane contour, on its two varying axes."""
    extents = np.ptp(points, axis=0)
    axis = int(np.argmin(extents))
    keep = [i for i in range(3) if i != axis]
    return axis, points[:, keep]


def _remove_collinear(poly2d: np.ndarray, rel_tol: float = 1e-3) -> np.ndarray:
    """Boolean mask dropping the points whose neighbours are (almost) on a
    straight line."""
    n = len(poly2d)
    if n < 3:
        return np.ones(n, dtype=bool)
    prev = np.roll(poly2d, 1, axis=0)
    nxt = np.roll(poly2d, -1, axis=0)
    a = prev - poly2d
    b = nxt - poly2d
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    keep = np.abs(cross) > rel_tol * (norm + 1e-12)
    if keep.sum() < 3:
        return np.ones(n, dtype=bool)
    return keep


def _ear_clip(poly2d: np.ndarray) -> list[tuple[int, int, int]] | None:
    """Triangulate a simple polygon (CCW 2D loop) by ear clipping.

    Unlike a Delaunay fill, the triangles never cross the contour, so
    concave features (e.g. the steps of a staircase-shaped part) are
    respected exactly instead of being bridged by a diagonal.
    """
    v = poly2d - poly2d[0]
    area2 = (v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]).sum()
    poly = poly2d[::-1] if area2 < 0 else poly2d
    n = len(poly)
    if n < 3:
        return None

    remaining = list(range(n))
    faces: list[tuple[int, int, int]] = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside_tri(t, p, orient):
        d1 = cross(t[0], t[1], p)
        d2 = cross(t[1], t[2], p)
        d3 = cross(t[2], t[0], p)
        if orient > 0:
            return d1 >= 0 and d2 >= 0 and d3 >= 0
        return d1 <= 0 and d2 <= 0 and d3 <= 0

    guard = 0
    while len(remaining) > 3:
        guard += 1
        if guard > 4 * n:
            return None
        clipped = False
        for k in range(len(remaining)):
            i_prev = remaining[k - 1]
            i_cur = remaining[k]
            i_nxt = remaining[(k + 1) % len(remaining)]

            tri = (poly[i_prev], poly[i_cur], poly[i_nxt])
            orient = cross(tri[0], tri[1], tri[2])
            if orient <= 0:
                continue

            others = [j for j in remaining if j not in (i_prev, i_cur, i_nxt)]
            if any(inside_tri(tri, poly[j], orient) for j in others):
                continue

            faces.append((i_prev, i_cur, i_nxt))
            remaining.pop(k)
            clipped = True
            break
        if not clipped:
            return None

    faces.append((remaining[0], remaining[1], remaining[2]))
    return faces


def _fill_loop(points: np.ndarray, tolerance: float = 1e-6) -> pv.PolyData | None:
    """Triangulate a plane contour into a filled patch, if it is a closed loop.

    The triangulation is constrained by the contour (ear clipping), so no
    triangle crosses into the part's volume: a staircase step stays a
    step instead of being bridged by a diagonal, which a Delaunay fill
    (delaunay_2d, projected on XY) would do on a vertical plane.
    """
    if len(points) < 3:
        return None
    scale = max(1.0, np.ptp(points, axis=0).max())
    if np.linalg.norm(points[0] - points[-1]) > tolerance * scale:
        return None

    loop = points[:-1]
    _, loop2d = _plane_coords(loop)
    keep = _remove_collinear(loop2d)
    if keep.sum() >= 3:
        loop = loop[keep]
        _, loop2d = _plane_coords(loop)
    if len(loop2d) < 3:
        return None

    faces = _ear_clip(loop2d)
    if faces is None:
        return None

    cells = np.hstack([[3, *f] for f in faces]).astype(np.int64)
    return pv.PolyData(np.asarray(loop, dtype=float), faces=cells)


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
