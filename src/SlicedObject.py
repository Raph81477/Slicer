"""Object-oriented slicer utilities for FDM printing path generation."""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


class SlicedObject:
    """
    Base class for a single sliced layer surface.

    Holds the geometric surface of one printing layer and provides the
    generic operations shared by every kind of sliced object (distance-to-
    boundary computation, offset surfaces, contour extraction, and
    toolpath construction). Subclasses (SlicedPiece, SlicedMold) add the
    behaviour that is specific to what is actually being printed.

    Performance notes
    -----------------
    Everything that only depends on the surface geometry (cell normals,
    distance-to-boundary field) is computed lazily and cached. The caches
    are dropped automatically whenever `surface` is reassigned.
    """

    def __init__(self, surface: pv.PolyData):
        self.surface = surface  # goes through the setter below

    # ------------------------------------------------------------------
    # Surface + cache management
    # ------------------------------------------------------------------

    @property
    def surface(self) -> pv.PolyData:
        return self._surface

    @surface.setter
    def surface(self, value: pv.PolyData) -> None:
        # Any geometry-dependent cache is invalid once the mesh changes
        # (e.g. after merge_points, which also changes point indexing).
        self._surface = value
        self._distance_tree_built = False
        self._cell_normals_cache = None

    @property
    def _cell_normals(self) -> np.ndarray:
        """
        Unit cell normals of the surface, computed once and cached.

        PyVista's `cell_normals` recomputes the normals of the WHOLE mesh
        on every access when no normals array is stored on it, so calling
        it once per path point is very expensive.
        """
        if self._cell_normals_cache is None:
            normals = np.array(self.surface.cell_normals, dtype=float)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            self._cell_normals_cache = normals / norms
        return self._cell_normals_cache

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def normal(self) -> np.ndarray:
        """Mean face normal of the layer surface (unit vector)."""
        n = self._cell_normals.mean(axis=0)
        return n / np.linalg.norm(n)

    def build_distance_tree(self) -> None:
        """
        Compute, for every point of the surface, the distance to the
        nearest boundary point, and store it as the "dist_to_boundary"
        scalar field on the surface.
        """
        edges = self.surface.extract_feature_edges()
        boundary_points = edges.points
        if len(boundary_points) == 0:
            raise ValueError("Surface has no boundary/feature edges: cannot compute dist_to_boundary.")

        tree = cKDTree(boundary_points)
        # workers=-1: use every available core for the nearest-neighbour query
        distances, _ = tree.query(self.surface.points, workers=-1)

        self.surface["dist_to_boundary"] = distances
        self._distance_tree_built = True

    def build_offset_surface(self, value: float, invert: bool = False) -> pv.PolyData:
        """
        Build a new surface whose boundary exactly matches the
        dist_to_boundary == value isoline.

        Note: this follows PyVista's clip_scalar convention, which is easy
        to get backwards: invert=True keeps the mesh BELOW value.

        invert=True : keep the zone dist_to_boundary <= value (border strip)
        invert=False (default): keep the zone dist_to_boundary >= value (inner zone)
        """
        if not self._distance_tree_built:
            self.build_distance_tree()

        # clip_scalar re-triangulates every cell crossed by "value", so the
        # resulting boundary lies exactly on the isovalue.
        return self.surface.clip_scalar(
            scalars="dist_to_boundary",
            value=value,
            invert=invert,
        )

    def extract_contour_polylines(
        self,
        isosurfaces,
        scalar_name: str = "dist_to_boundary",
    ) -> list[np.ndarray]:
        """
        Generate the distance-to-boundary isolines and return them as a
        list of separate, ordered polylines (instead of a single PolyData).
        """
        contour_poly = self.surface.contour(isosurfaces=isosurfaces, scalars=scalar_name)

        if contour_poly.n_points == 0:
            return []

        return self._polylines_from_polydata(contour_poly)

    def get_local_normals(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_local_normal: unit surface normal at the
        closest cell for each row of `points` (shape (n, 3)).

        A single batched closest-cell query builds the spatial locator once
        instead of once per point.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        cell_ids = np.atleast_1d(self.surface.find_closest_cell(points))
        return self._cell_normals[cell_ids]

    def get_local_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Get the local surface normal at the closest point of the surface
        (used for perpendicular approach/leave moves).

        Subclasses may override this method to change the normal
        definition; the batched helpers below honour such an override.
        """
        return self.get_local_normals(np.asarray(point, dtype=float)[None, :])[0]

    def _normals_at(self, points: np.ndarray) -> np.ndarray:
        """
        Local normals for many points. Uses the fast batched query unless a
        subclass overrides get_local_normal, in which case the override is
        called point by point so its behaviour is preserved.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if type(self).get_local_normal is not SlicedObject.get_local_normal:
            return np.array([self.get_local_normal(p) for p in points])
        return self.get_local_normals(points)

    # ------------------------------------------------------------------
    # Polyline utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _polylines_from_polydata(poly: pv.PolyData) -> list[np.ndarray]:
        """Split a stripped PolyData's `lines` connectivity into a list of point arrays."""
        stripped = poly.strip(join=True)

        polylines = []
        pts = stripped.points
        lines = stripped.lines

        i = 0
        n_total = len(lines)
        while i < n_total:
            n = lines[i]
            if n >= 2:
                polylines.append(pts[lines[i + 1 : i + 1 + n]])
            i += n + 1

        return polylines

    @classmethod
    def order_segments(cls, cut: pv.PolyData) -> list[np.ndarray]:
        """
        Reconstruct ordered polylines from a pv.PolyData of unordered
        segments, using vtkStripper (natively handles multiple connected
        components and junctions).
        """
        if cut.n_points == 0:
            return []

        # vtkStripper merges contiguous segments into polylines while
        # respecting separate connected components.
        return cls._polylines_from_polydata(cut)

    # ------------------------------------------------------------------
    # Toolpath construction
    # ------------------------------------------------------------------

    def build_path_single_pass(
        self,
        curves: list[np.ndarray],
        approach_code: int,
        connection_end_code: int,
        continuation_code: int,
        approach_distance: float = 1.0,
        max_connection_length: float = None,
        start_curve_idx: int = 0,
        start_at_end: bool = False,
        connection_blocker=None,
    ):
        """
        Stitch a list of curves into a single, fully connected pass.

        If a connection segment C_con would exceed max_connection_length,
        it is turned into a sub-path end (leave move) followed by a new
        sub-path start (approach move) instead of being kept as a
        continuous connection.

        connection_blocker: optional callable (p_start, p_end) -> bool. When
        it returns True the connection p_start -> p_end is forbidden (e.g.
        it would re-print a crossing) and is turned into a leave + approach
        move, exactly as if it were longer than max_connection_length.

        Performance: the curve endpoints are gathered once into a single
        array and the nearest unused endpoint is found with a masked,
        vectorized distance computation. The previous implementation
        rebuilt a KD-tree from scratch at every iteration. Surface normals
        for approach/leave moves are also computed in one batched query.
        """
        curves = [np.asarray(c) for c in curves if len(c) >= 2]
        n_curves = len(curves)
        if n_curves == 0:
            return [], [], []

        # Endpoint table: row 2*k is the first point of curve k, row 2*k+1
        # its last point.
        endpoints = np.empty((2 * n_curves, 3))
        for k, c in enumerate(curves):
            endpoints[2 * k] = c[0]
            endpoints[2 * k + 1] = c[-1]
        endpoint_active = np.ones(2 * n_curves, dtype=bool)

        # Normals at every endpoint, computed once (batched).
        endpoint_normals = self._normals_at(endpoints)

        Lcmd = []
        path_segments = []
        segment_types = []

        def start_new_subpath(p_start, normal):
            origin = p_start + approach_distance * normal
            path_segments.append(np.array([origin, p_start]))
            segment_types.append("approach")
            Lcmd.append((p_start, approach_code))

        def end_subpath(p_end, normal):
            target = p_end + approach_distance * normal
            path_segments.append(np.array([p_end, target]))
            segment_types.append("leave")

        def consume(curve_idx, reverse):
            """Deactivate a curve; return it oriented + the endpoint-table rows of its first/last point."""
            endpoint_active[2 * curve_idx : 2 * curve_idx + 2] = False
            chosen = curves[curve_idx]
            if reverse:
                chosen = chosen[::-1]
            first_row = 2 * curve_idx + (1 if reverse else 0)
            last_row = 2 * curve_idx + (0 if reverse else 1)
            return chosen, first_row, last_row

        # --- first curve -------------------------------------------------
        first, first_row, last_row = consume(start_curve_idx, start_at_end)
        start_new_subpath(first[0], endpoint_normals[first_row])

        path_segments.append(first)
        segment_types.append("curve")
        Lcmd.append((first[-1], continuation_code))
        p_start = first[-1]
        p_start_row = last_row
        n_remaining = n_curves - 1

        # --- chain the remaining curves ----------------------------------
        while n_remaining > 0:
            d2 = np.einsum("ij,ij->i", endpoints - p_start, endpoints - p_start)
            d2[~endpoint_active] = np.inf
            idx = int(np.argmin(d2))
            dist = np.sqrt(d2[idx])
            curve_idx, is_end = idx // 2, bool(idx % 2)
            p_end = endpoints[idx]

            chosen, first_row, last_row = consume(curve_idx, is_end)
            n_remaining -= 1

            too_long = max_connection_length is not None and dist > max_connection_length
            blocked = (not too_long) and connection_blocker is not None \
                and connection_blocker(p_start, p_end)

            if too_long or blocked:
                # Genuine topological gap: properly close this sub-path
                # and start a new one, instead of forcing a huge connection.
                end_subpath(p_start, endpoint_normals[p_start_row])
                start_new_subpath(chosen[0], endpoint_normals[first_row])

                path_segments.append(chosen)
                segment_types.append("curve")
                Lcmd.append((chosen[-1], continuation_code))
            else:
                # Normal case: legitimate connection
                Lcmd.append((p_end, connection_end_code))
                path_segments.append(np.array([p_start, p_end]))
                segment_types.append("connection")

                path_segments.append(chosen)
                segment_types.append("curve")
                Lcmd.append((chosen[-1], continuation_code))

            p_start = chosen[-1]
            p_start_row = last_row

        end_subpath(p_start, endpoint_normals[p_start_row])

        return path_segments, Lcmd, segment_types

    def build_full_printing_path(
        self,
        pure_curves: list[np.ndarray],
        composite_curves: list[np.ndarray],
        approach_distance: float = 1.0,
        max_connection_length: float = None,
    ):
        """
        Full path-planning algorithm: two separate passes (pure, then
        composite), each fully connected internally.
        """
        pure_segments, pure_cmd, pure_types = self.build_path_single_pass(
            pure_curves,
            approach_code=5,
            connection_end_code=3,
            continuation_code=2,
            approach_distance=approach_distance,
            max_connection_length=max_connection_length,
        )

        composite_segments, composite_cmd, composite_types = self.build_path_single_pass(
            composite_curves,
            approach_code=4,
            connection_end_code=1,
            continuation_code=0,
            approach_distance=approach_distance,
            max_connection_length=max_connection_length,
        )

        all_segments = pure_segments + composite_segments
        Lcmd = pure_cmd + composite_cmd
        segment_types = pure_types + composite_types

        return all_segments, Lcmd, segment_types

    # ------------------------------------------------------------------
    # Path export (for external trajectory simulation, e.g. RoboDK)
    # ------------------------------------------------------------------

    def compute_path_normals(self, path_segments: list[np.ndarray]) -> list[np.ndarray]:
        """
        For every point of every segment, compute the local surface normal
        (see get_local_normal, overridden by subclasses). This is the
        natural approach axis of a deposition tool (perpendicular to the
        surface).

        All points of all segments are queried in a single batch, which is
        much faster than one closest-cell lookup per point.
        """
        if not path_segments:
            return []

        lengths = [len(seg) for seg in path_segments]
        all_points = np.vstack([np.asarray(seg, dtype=float) for seg in path_segments])
        all_normals = self._normals_at(all_points)

        return np.split(all_normals, np.cumsum(lengths)[:-1])

    @staticmethod
    def simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
        """
        Simplify a polyline with the Ramer-Douglas-Peucker algorithm:
        keeps the points needed to stay within `tolerance` (the maximum
        allowed perpendicular distance from the original polyline) and
        drops the rest. Always keeps the first and last point.

        Works directly in 3D (the perpendicular distance is computed to
        the 3D chord between the two current endpoints), and uses an
        iterative stack instead of recursion to stay safe on long paths.
        """
        points = np.asarray(points, dtype=float)
        n = len(points)
        if n < 3:
            return points

        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        keep[-1] = True

        stack = [(0, n - 1)]

        while stack:
            start_idx, end_idx = stack.pop()
            if end_idx <= start_idx + 1:
                continue

            a = points[start_idx]
            b = points[end_idx]
            chunk = points[start_idx + 1 : end_idx]

            segment = b - a
            seg_len_sq = np.dot(segment, segment)

            if seg_len_sq < 1e-12:
                # a and b coincide: perpendicular distance collapses to
                # the plain distance to a.
                dists = np.linalg.norm(chunk - a, axis=1)
            else:
                t = (chunk - a) @ segment / seg_len_sq
                proj = a + t[:, None] * segment
                dists = np.linalg.norm(chunk - proj, axis=1)

            max_idx_local = np.argmax(dists)
            max_dist = dists[max_idx_local]

            if max_dist > tolerance:
                split_idx = start_idx + 1 + max_idx_local
                keep[split_idx] = True
                stack.append((start_idx, split_idx))
                stack.append((split_idx, end_idx))

        return points[keep]

    @staticmethod
    def compute_path_tangents(path_segments: list[np.ndarray]) -> list[np.ndarray]:
        """
        For every point of every segment, compute the unit tangent to the
        trajectory (direction of travel): a central difference for
        interior points, a one-sided difference at each segment's two
        endpoints. This is the natural travel axis of a deposition tool.

        Note: tangents are computed independently within each segment, so
        there can be a small discontinuity right at a segment boundary
        (e.g. curve -> connection) - this only affects the tool's rotation
        about its approach axis at that single point, not its position.
        """
        tangents = []

        for seg in path_segments:
            seg = np.asarray(seg, dtype=float)
            n = len(seg)

            if n == 1:
                # Degenerate single-point segment: direction is undefined,
                # any unit vector will be re-orthogonalized later anyway.
                tangents.append(np.array([[1.0, 0.0, 0.0]]))
                continue

            tan = np.empty_like(seg)
            tan[1:-1] = seg[2:] - seg[:-2]
            tan[0] = seg[1] - seg[0]
            tan[-1] = seg[-1] - seg[-2]

            norms = np.linalg.norm(tan, axis=1)
            norms[norms < 1e-12] = 1.0
            tangents.append(tan / norms[:, None])

        return tangents

    @staticmethod
    def build_orthonormal_frames(
        tangents: list[np.ndarray],
        normals: list[np.ndarray],
        approach_sign: float = -1.0,
    ) -> list[np.ndarray]:
        """
        Combine, point by point, a travel-direction tangent and a surface
        normal into a full right-handed orthonormal tool frame - i.e. the
        rotation part of a robot pose.

        - z_axis = approach_sign * normal. approach_sign=-1 (default)
          points the tool's Z axis INTO the surface (the usual "approach"
          convention); use +1 if your tool/robot convention points the
          Z axis away from the surface instead.
        - x_axis = the tangent, re-orthogonalized against z_axis via
          Gram-Schmidt, so it always represents "direction of travel"
          even where the raw tangent isn't perfectly perpendicular to the
          normal. Where the tangent is nearly parallel to z_axis (e.g. a
          straight approach/leave move, which travels along the normal
          itself), an arbitrary perpendicular fallback is used instead,
          since travel direction carries no meaningful orientation there.
        - y_axis = z_axis x x_axis, so that (x_axis, y_axis, z_axis) is a
          right-handed basis.

        Because the tool is NOT rotationally symmetric about its approach
        axis, this x_axis/y_axis choice matters mechanically, not just
        for bookkeeping: it fixes the tool's rotation about its own axis,
        so getting it wrong can mean the tool colliding with the part.

        Returns one array per segment, of shape (n_points, 3, 3), where
        frame[:, :, 0] is x_axis, frame[:, :, 1] is y_axis and
        frame[:, :, 2] is z_axis - i.e. each frame[i] is a rotation
        matrix whose columns are the tool's X, Y, Z axes in world
        coordinates (the rotation part of a robot pose).

        Performance: all points of all segments are processed together
        with array operations (no per-point Python loop).
        """
        if len(tangents) == 0:
            return []

        lengths = [len(t) for t in tangents]
        t = np.vstack(tangents).astype(float)
        z = approach_sign * np.vstack(normals).astype(float)
        z /= np.linalg.norm(z, axis=1, keepdims=True)

        # Gram-Schmidt: remove the component of the tangent along z
        x = t - np.sum(t * z, axis=1, keepdims=True) * z
        x_norm = np.linalg.norm(x, axis=1)

        # Tangent nearly parallel to z (straight approach/leave move):
        # fall back to an arbitrary vector perpendicular to z.
        degenerate = x_norm < 1e-8
        if np.any(degenerate):
            fallback = np.zeros_like(z)
            fallback[:, 0] = 1.0
            use_y = np.abs(z[:, 0]) > 0.9
            fallback[use_y] = [0.0, 1.0, 0.0]

            x_fb = fallback - np.sum(fallback * z, axis=1, keepdims=True) * z
            x[degenerate] = x_fb[degenerate]
            x_norm[degenerate] = np.linalg.norm(x_fb[degenerate], axis=1)

        x /= x_norm[:, None]
        y = np.cross(z, x)

        # frames[i, :, 0] = x, frames[i, :, 1] = y, frames[i, :, 2] = z
        frames = np.stack([x, y, z], axis=2)

        return np.split(frames, np.cumsum(lengths)[:-1])

    def compute_path_orientation(
        self,
        path_segments: list[np.ndarray],
        approach_sign: float = -1.0,
    ) -> list[np.ndarray]:
        """
        Convenience wrapper: compute the full tool orientation (rotation
        matrix per point, see build_orthonormal_frames) for this layer's
        path in one call.
        """
        normals = self.compute_path_normals(path_segments)
        tangents = self.compute_path_tangents(path_segments)
        return self.build_orthonormal_frames(tangents, normals, approach_sign=approach_sign)

    @staticmethod
    def export_path_to_txt(
        path_segments: list[np.ndarray],
        segment_types: list[str],
        filepath,
        orientations: list[np.ndarray] | None = None,
        layer_id=None,
        mode: str = "w",
    ) -> None:
        """
        Write an ordered printing path to a plain-text file, one point per
        line: "x y z [ux uy uz vx vy vz wx wy wz] type [layer]", where
        (u, v, w) are the tool's X, Y, Z axes in world coordinates - i.e.
        a full robot pose, position plus orientation.

        - path_segments / segment_types: as returned by
          build_full_printing_path (SlicedPiece) or build_support_path
          (SlicedMold).
        - orientations: optional, same shape as path_segments, one 3x3
          rotation matrix per point (see compute_path_orientation /
          build_orthonormal_frames). When given, each point's tool X, Y,
          Z axes are written right after its coordinates, ready to build
          a 4x4 pose matrix on the receiving end (e.g. in RoboDK).
        - layer_id: optional label (e.g. a layer index) written as a
          trailing column; handy when appending several layers to the
          same file so they stay distinguishable.
        - mode: "w" (default) (re)creates the file and writes a header;
          "a" appends rows to an existing file without rewriting the
          header. Use "a" from the second layer onward when building up
          a multi-layer trajectory file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_header = mode == "w" or not filepath.exists()

        # One printf-style format per segment: the (constant) type and layer
        # labels are baked into the format, so each segment is written with
        # a single np.savetxt call instead of one f-string per point.
        numeric_fmt = ["%.4f"] * 3
        if orientations is not None:
            numeric_fmt += ["%.6f"] * 9
        layer_suffix = f" {layer_id}" if layer_id is not None else ""

        with open(filepath, mode, encoding="utf-8", newline="") as f:
            if write_header:
                header = ["x", "y", "z"]
                if orientations is not None:
                    header += ["ux", "uy", "uz", "vx", "vy", "vz", "wx", "wy", "wz"]
                header.append("type")
                if layer_id is not None:
                    header.append("layer")
                f.write(" ".join(header) + "\n")

            for seg_idx, (seg, seg_type) in enumerate(zip(path_segments, segment_types)):
                seg = np.asarray(seg, dtype=float)
                if len(seg) == 0:
                    continue

                if orientations is not None:
                    # Per point: X axis (3), Y axis (3), Z axis (3), i.e.
                    # the columns of the rotation matrix, in this order.
                    R = np.asarray(orientations[seg_idx])
                    data = np.hstack([seg, R.transpose(0, 2, 1).reshape(len(seg), 9)])
                else:
                    data = seg

                fmt = " ".join(numeric_fmt) + f" {seg_type}{layer_suffix}"
                np.savetxt(f, data, fmt=fmt, newline="\n")

    def export_printing_path(
        self,
        path_segments: list[np.ndarray],
        segment_types: list[str],
        filepath,
        include_orientation: bool = False,
        approach_sign: float = -1.0,
        simplify_tolerance: float | None = None,
        layer_id=None,
        mode: str = "w",
    ) -> None:
        """
        Convenience wrapper around export_path_to_txt: optionally
        simplifies each segment with Ramer-Douglas-Peucker (see
        simplify_polyline) to shrink the exported file/robot program,
        then optionally computes this layer's full tool orientation
        (normal + travel tangent, see compute_path_orientation) - on the
        simplified points, so the orientation written out always matches
        the points actually written out.

        Note: simplification is applied independently to each segment
        (curve, connection, approach, leave), so it only removes points
        along a single straight/curved stretch - it never merges distinct
        segments together or drops a segment's two endpoints.
        """
        if simplify_tolerance is not None:
            path_segments = [
                self.simplify_polyline(seg, simplify_tolerance) for seg in path_segments
            ]

        path_segments = self.drop_shared_points(path_segments)

        orientations = (
            self.compute_path_orientation(path_segments, approach_sign=approach_sign)
            if include_orientation
            else None
        )

        self.export_path_to_txt(
            path_segments,
            segment_types,
            filepath,
            orientations=orientations,
            layer_id=layer_id,
            mode=mode,
        )

    @staticmethod
    def drop_shared_points(segments, atol=1e-6):
        out, last = [], None
        for seg in segments:
            seg = np.asarray(seg, dtype=float)
            if last is not None and len(seg) and np.allclose(seg[0], last, atol=atol):
                seg = seg[1:]
            out.append(seg)
            last = out[-1][-1] if len(out[-1]) else last
        return out
