# Non-planar slicer (composite 3D printing, 6-DOF robotic arm)

Non-planar slicing of composite models (continuous fibers) for printing
with a 6-degree-of-freedom robotic arm. Trajectory generation is done in
Python (PyVista / NumPy / SciPy); the robot simulation is done in
RoboDK.

## Structure

- `src/slicer/core/SlicedObject.py` — parent class (layer surface,
  distance to the edge, offset, contours, path construction, trajectory
  export).
- `src/slicer/core/SlicedPiece.py` — derived class for the piece to
  print (fibers, fiber-aligned slicing planes, fiber-aligned path +
  contours).
- `src/slicer/core/SlicedMold.py` — derived class for the
  automatically generated bottom mold (shadow volume, slicing, zigzag
  infill, export).
- `src/slicer/utils/visualization.py` — visualization of the mold by
  slicing planes (filled patches, ear-clipping).
- `src/slicer/gui/` — Tkinter graphical interface.

## GUI (Tkinter)

Launch the interface:

```bash
python -m slicer.gui
```

Features:

- Selection of the .dat file (ANSYS model) and the fiber folder
  (CSV files exported by OAMC).
- Manual entry of `layer_height` and `layer_width` (nozzle diameter),
  plus `fiber_diameter` and `n_layers_under_piece`.
- "Rotate and lift model" and "Mold generation" options. Enabling the
  mold automatically enables rotate and lift. The lift positions the
  model such that `zmin > 0`, then raises it by
  `n_layers_under_piece * layer_height`. When the mold is enabled, the
  folder where the transformed fibers (rotation + lift) are saved can
  be selected.
- Tunable path parameters for the piece: `contour_width`,
  `approach_distance`, `max_connection_length`, `merge_tolerance`,
  `contour_isovalues`.
- Tunable edge-extraction parameters (used for contour computation):
  `feature_angle`, `boundary_edges`, `non_manifold_edges`.
- Mold parameters: shadow volume method (vertical / cone / simple),
  `infill_percentage`, `n_contour_shells`.
- Visualization in an external PyVista figure, with separate options:
  model, detected edges, print path (blue curves, red connections,
  green approach, orange leave), mold (visualization by slicing
  planes), legend, and selection of layers to display (e.g. `0-4` or
  `2,5,7`).
- Materials (transversely isotropic fiber + isotropic matrix)
  editable.
- Export of trajectories (piece + mold) to text files ready for
  RoboDK.

- "Fiber tuning" tab: complete OAMC pipeline for fiber
  generation/optimization (bearing load, optional mold, least-squares
  optimization with iterations, LPP load paths, RDP/outlier
  post-processing, fiber and convergence CSV export), reused from the
  standalone OAMC tuner, in a dedicated page of the slicer.
