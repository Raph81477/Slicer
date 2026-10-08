import csv
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as Rot

from oamc.core import CompositeModel
from oamc.utils.vtk import compute_level_surface
from SlicedObject import SlicedObject


class SlicedPiece(SlicedObject):
    """
    A sliced layer of the printed composite piece.

    In addition to the generic slicing/toolpath behaviour inherited from
    SlicedObject, a SlicedPiece knows about the fiber paths placed inside
    it and can generate the infill slicing planes aligned with the local
    fiber direction.
    """

    def __init__(self, surface: pv.PolyData, fiber_paths: dict[int, np.ndarray] | None = None):
        super().__init__(surface)
        self.fiber_paths = fiber_paths or {}

    # ------------------------------------------------------------------
    # Fiber data
    # ------------------------------------------------------------------

    @property
    def fiber_paths(self) -> dict[int, np.ndarray]:
        return self._fiber_paths

    @fiber_paths.setter
    def fiber_paths(self, value: dict[int, np.ndarray]) -> None:
        # The cached mean direction depends on the fiber paths.
        self._fiber_paths = value
        self._fiber_mean_direction_cache = None

    @staticmethod
    def load_fiber_paths(directory: Path, layer: int = 1) -> dict[int, np.ndarray]:
        """
        Load the fiber paths of a given layer from the
        "fibers/layer_<layer>" CSV files.

        TODO: currently only extracts a single layer at a time.
        """
        layer_dir = Path(directory) / f"fibers/layer_{layer}"

        fiber_paths = {}

        for i, csv_file in enumerate(sorted(layer_dir.glob(f"fiber_{layer}_*.csv")), start=1):
            with open(csv_file, mode="r", encoding="utf-8", newline="") as file:
                raw_path = np.array(list(csv.reader(file)), dtype=float)

            fiber_paths[i] = raw_path[:, :3]

        return fiber_paths

    @property
    def fiber_mean_direction(self) -> np.ndarray:
        """
        Principal direction of the fiber paths (first principal component
        of the normalized segment directions).

        Computed once per instance and cached. Segment directions are
        taken within each fiber, so no spurious segment is created between
        the last point of one fiber and the first point of the next.
        """
        if self._fiber_mean_direction_cache is None:
            self._fiber_mean_direction_cache = self._compute_fiber_mean_direction(
                self.fiber_paths
            )
        return self._fiber_mean_direction_cache

    @staticmethod
    def _compute_fiber_mean_direction(fiber_paths: dict[int, np.ndarray]) -> np.ndarray:
        dirs = []
        for path in fiber_paths.values():
            segs = np.diff(path, axis=0)
            norms = np.linalg.norm(segs, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-12  # skip zero-length (duplicate) points
            dirs.append(segs[valid] / norms[valid])

        all_dirs = np.vstack(dirs)

        # Principal axis of the direction bundle: dominant eigenvector of
        # the (uncentered) second-moment matrix. Centering would make the
        # PCA return the axis of greatest SCATTER, i.e. a direction
        # perpendicular to a bundle of nearly parallel fibers.
        _, eigvecs = np.linalg.eigh(all_dirs.T @ all_dirs)
        principal = eigvecs[:, -1]

        # Deterministic sign: make the largest-magnitude component positive.
        if principal[np.argmax(np.abs(principal))] < 0:
            principal = -principal

        return principal

    # ------------------------------------------------------------------
    # Slicing planes
    # ------------------------------------------------------------------

    def generate_slicing_planes(self, i_surface: int = 0, spacing: float = 0.5):
        """
        Generate the infill slicing planes for this layer, alternating the
        45-degree offset direction (relative to the mean fiber direction)
        every other layer. `spacing` is the distance between two planes.
        """
        mean_dir = self.fiber_mean_direction
        n = self.normal

        sign = 1.0 if i_surface % 2 == 0 else -1.0
        direction = (
            np.cos(np.deg2rad(45)) * mean_dir
            + sign * np.sin(np.deg2rad(45)) * np.cross(n, mean_dir)
        )
        direction /= np.linalg.norm(direction)

        # Slicing plane normal
        m = np.cross(n, direction)
        m /= np.linalg.norm(m)

        pts = self.surface.points
        proj = pts @ m

        orig = pts.mean(axis=0)

        # Projection RELATIVE to the reference point used to build the planes
        proj = (pts - orig) @ m
        smin, smax = proj.min(), proj.max()

        # Small margin so that the extreme planes fully bracket the surface
        margin = 0.5 * spacing
        smin -= margin
        smax += margin

        # Center the plane family on the extent so both ends are covered evenly
        n_planes = int(np.ceil((smax - smin) / spacing)) + 1
        positions = smin + spacing * np.arange(n_planes)

        return [
            {"origin": orig + s * m, "normal": m}
            for s in positions
        ]

    @staticmethod
    def _chain_segments(cut: pv.PolyData) -> list[np.ndarray] | None:
        """
        Chain the 2-point line cells of `cut` into ordered polylines.

        vtkStripper turned out to leave the output of a multi-level contour
        fragmented, so the (simple) chaining is done here. Returns None if
        the segment graph is not made of simple chains/loops (a point shared
        by more than two segments), so that the caller can fall back to a
        slower but more general method.
        """
        cut = cut.clean()  # merge coincident points
        lines = np.asarray(cut.lines)
        if lines.size == 0 or lines.size % 3 or not np.all(lines[::3] == 2):
            return None

        segs = lines.reshape(-1, 3)[:, 1:]
        n_pts = cut.n_points

        degree = np.bincount(segs.ravel(), minlength=n_pts)
        if degree.max() > 2:
            return None

        # Each point has at most two neighbours (-1 = none)
        neighbors = np.full((n_pts, 2), -1, dtype=np.int64)
        fill = np.zeros(n_pts, dtype=np.int64)
        for a, b in segs:
            neighbors[a, fill[a]] = b
            fill[a] += 1
            neighbors[b, fill[b]] = a
            fill[b] += 1

        visited = np.zeros(n_pts, dtype=bool)
        chains = []

        def walk(start):
            chain = [start]
            visited[start] = True
            prev, cur = -1, start
            while True:
                nxt = -1
                for cand in neighbors[cur]:
                    if cand != -1 and cand != prev and not visited[cand]:
                        nxt = cand
                        break
                if nxt == -1:
                    break
                chain.append(nxt)
                visited[nxt] = True
                prev, cur = cur, nxt
            return chain

        # Open chains first (start from their end points), then closed loops
        for start in np.flatnonzero(degree == 1):
            if not visited[start]:
                chains.append(walk(start))
        for start in np.flatnonzero((degree == 2) & ~visited):
            if not visited[start]:
                chain = walk(start)
                chain.append(chain[0])  # close the loop
                chains.append(chain)

        pts = cut.points
        return [pts[c] for c in chains if len(c) >= 2]

    @classmethod
    def _cut_with_planes(cls, zone: pv.PolyData, planes: list[dict]) -> list[np.ndarray]:
        """
        Cut `zone` with ALL the (parallel) slicing planes in one pass and
        return the resulting polylines, grouped plane by plane.

        Calling `zone.slice` once per plane traverses the whole mesh for
        every plane. Since the planes are parallel, they are level sets of
        a single scalar field s(p) = p . m, so one contour filter call
        with all the levels produces the same lines in a single pass.
        """
        if not planes or zone.n_points == 0:
            return []

        m = planes[0]["normal"]
        levels = np.array([plane["origin"] @ m for plane in planes])

        zone = zone.copy(deep=False)
        zone.point_data["_plane_offset"] = zone.points @ m

        cut = zone.contour(isosurfaces=levels.tolist(), scalars="_plane_offset")
        if cut.n_points == 0:
            return []

        polylines = cls._chain_segments(cut)
        if polylines is None:
            return cls._cut_with_planes_slow(zone, planes)
        if not polylines:
            return []

        # The contour filter does not guarantee a plane-by-plane output
        # order: restore it so the toolpath is the same as with per-plane
        # slicing. All points of a polyline lie on one plane.
        offsets = np.array([poly[0] @ m for poly in polylines])
        plane_idx = np.abs(offsets[:, None] - levels[None, :]).argmin(axis=1)
        order = np.argsort(plane_idx, kind="stable")

        return [polylines[i] for i in order]

    @classmethod
    def _cut_with_planes_slow(cls, zone: pv.PolyData, planes: list[dict]) -> list[np.ndarray]:
        """Reference implementation: one `slice` call per plane (fallback)."""
        lines = []
        for plane in planes:
            cut = zone.slice(normal=plane["normal"], origin=plane["origin"])
            if cut.n_points > 0:
                lines.extend(cls.order_segments(cut))
        return lines

    # ------------------------------------------------------------------
    # Construction from a CompositeModel
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_levels(model: CompositeModel):
        """Return (grid with the "q" field, array of isovalues, one per layer)."""
        if not np.any(model.q):
            model.init_q()
        max_q_value = np.max(model.q)

        grid = model.get_grid(u_scaling_factor=0)
        grid.point_data["q"] = model.q

        q_values = np.arange(1.0, int(max_q_value))  # Layers 1 to N
        return grid, q_values

    @staticmethod
    def get_printing_layers(
        model: CompositeModel,
        layer_indices: list[int] | None = None,
    ) -> list[pv.PolyData]:
        """
        Compute the raw isosurface geometry for each printing layer of the
        model. When `layer_indices` is given, only those layers are
        extracted (the isosurface extraction is the expensive part).
        """
        grid, q_values = SlicedPiece._layer_levels(model)

        if layer_indices is not None:
            q_values = q_values[list(layer_indices)]

        return [compute_level_surface(grid, name="q", level=q_val) for q_val in q_values]

    @classmethod
    def slice_model(
        cls,
        model: CompositeModel,
        fiber_paths: dict[int, np.ndarray] | None = None,
    ) -> list["SlicedPiece"]:
        """Slice a CompositeModel into a list of SlicedPiece layers."""
        surfaces = cls.get_printing_layers(model)
        pieces = [cls(surface, fiber_paths) for surface in surfaces]

        # The fiber paths are shared by all the layers: compute their mean
        # direction once instead of once per layer.
        if pieces and fiber_paths:
            direction = pieces[0].fiber_mean_direction
            for piece in pieces[1:]:
                piece._fiber_mean_direction_cache = direction

        return pieces

    @classmethod
    def from_model(
        cls,
        model: CompositeModel,
        i_surface: int = 0,
        fiber_paths: dict[int, np.ndarray] | None = None,
    ) -> "SlicedPiece":
        """
        Build a single SlicedPiece layer from a CompositeModel, by layer
        index. Equivalent to
        `SlicedPiece.slice_model(model, fiber_paths)[i_surface]`, but only
        the requested isosurface is extracted.
        """
        grid, q_values = cls._layer_levels(model)

        if not 0 <= i_surface < len(q_values):
            raise IndexError(
                f"i_surface={i_surface} out of range: model has {len(q_values)} layers"
            )

        surface = compute_level_surface(grid, name="q", level=q_values[i_surface])
        return cls(surface, fiber_paths=fiber_paths)

    @staticmethod
    def get_mean_surface_normals(
        model: CompositeModel,
        surfaces: list[pv.PolyData] | None = None,
    ) -> np.ndarray:
        """
        Mean of all layer face normals over the whole model.

        Pass `surfaces` (e.g. from get_printing_layers) to avoid
        recomputing the isosurfaces.
        """
        if surfaces is None:
            surfaces = SlicedPiece.get_printing_layers(model)

        normal_sum = np.zeros(3)
        n_cells = 0
        for surface in surfaces:
            surface.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            normals = surface.cell_data["Normals"]
            normal_sum += normals.sum(axis=0)
            n_cells += len(normals)

        return normal_sum / n_cells

    @staticmethod
    def get_rotation_matrix(
        model: CompositeModel,
        surfaces: list[pv.PolyData] | None = None,
    ) -> np.ndarray:
        """
        Rotation matrix aligning the mean layer normal of the model with
        the global Z axis.
        """
        target = np.array([0, 0, 1])
        mean_normal = -1 * SlicedPiece.get_mean_surface_normals(model, surfaces)
        axis = np.cross(mean_normal, target)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            R = np.eye(3)
        else:
            axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(mean_normal, target), -1.0, 1.0))
            R = Rot.from_rotvec(angle * axis).as_matrix()

        return R

    # ------------------------------------------------------------------
    # Full per-layer printing path (fiber-aligned infill + contour shells)
    # ------------------------------------------------------------------

    def build_printing_path(
        self,
        i_surface: int = 0,
        contour_width: float = 3.0,
        approach_distance: float = 1.0,
        max_connection_length: float = 2.0,
        contour_isovalues=(0.5, 1.0, 1.5, 2.0, 2.5),
        merge_tolerance: float = 0.1,
    ):
        """
        Build the full printing path for this piece layer: fiber-aligned
        infill lines (cut from the inner offset zone by the fiber-aligned
        slicing planes) plus distance-to-boundary contour shells, stitched
        together as in build_full_printing_path.
        """
        # Merge coincident points first: an isosurface extracted from the
        # model's grid can have duplicate points along its boundary, which
        # confuses feature-edge (boundary) detection and makes
        # build_offset_surface / extract_contour_polylines unreliable.
        # Assigning self.surface also invalidates every cached quantity
        # (normals, distance field), since the point indexing changes.
        self.surface = self.surface.merge_points(tolerance=merge_tolerance)

        planes = self.generate_slicing_planes(i_surface=i_surface)
        inner_zone = self.build_offset_surface(contour_width, invert=False)

        # Single-pass cut of the inner zone with all the planes
        infill_lines = self._cut_with_planes(inner_zone, planes)

        contour_lines = self.extract_contour_polylines(
            isosurfaces=list(contour_isovalues),
            scalar_name="dist_to_boundary",
        )

        return self.build_full_printing_path(
            pure_curves=contour_lines,
            composite_curves=infill_lines,
            approach_distance=approach_distance,
            max_connection_length=max_connection_length,
        )

    @classmethod
    def export_layers_to_txt(
        cls,
        pieces: list["SlicedPiece"],
        filepath,
        contour_width: float = 3.0,
        approach_distance: float = 1.0,
        max_connection_length: float = 2.0,
        contour_isovalues=(0.5, 1.0, 1.5, 2.0, 2.5),
        include_orientation: bool = False,
        approach_sign: float = -1.0,
        simplify_tolerance: float | None = None,
    ) -> None:
        """
        Build the full printing path for every given piece layer and
        write the whole trajectory (all layers, in order) to a single
        text file, ready to be loaded by an external trajectory simulator
        such as RoboDK.

        simplify_tolerance: when given, each segment is simplified with
        Ramer-Douglas-Peucker (see SlicedObject.simplify_polyline) before
        being written out, to shrink the exported file/robot program.
        """
        filepath = Path(filepath)

        for i, piece in enumerate(pieces):
            segments, _, types = piece.build_printing_path(
                i_surface=i,
                contour_width=contour_width,
                approach_distance=approach_distance,
                max_connection_length=max_connection_length,
                contour_isovalues=contour_isovalues,
            )

            piece.export_printing_path(
                segments,
                types,
                filepath,
                include_orientation=include_orientation,
                approach_sign=approach_sign,
                simplify_tolerance=simplify_tolerance,
                layer_id=i,
                mode="w" if i == 0 else "a",
            )

        print(f"Piece path exported to {filepath} ({len(pieces)} layers)")
