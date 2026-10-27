"""Tkinter front-end for the non-planar slicer.

The window collects the slicer inputs (fiber files, .dat model, printing
and path parameters) and runs the slicing in a worker thread so the GUI
stays responsive; the PyVista visualization opens in an external,
off-screen-independent figure once the computation is done.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, BOTTOM, END, LEFT, RIGHT, TOP, X, filedialog, messagebox, ttk

import numpy as np

from slicer.core import SlicedMold, SlicedPiece

try:
    import pyvista as pv
    from slicer.utils.visualization import build_slicing_plane_patches
except Exception:  # pragma: no cover - pyvista is a hard dependency anyway
    pv = None


DEFAULTS = {
    "E1": 66550.0,
    "E2": 4900.0,
    "nu12": 0.32,
    "G12": 1880.0,
    "G23": 1680.0,
    "E_matrix": 2990.0,
    "nu_matrix": 0.39,
    "fiber_dir": "fibers",
    "transformed_fiber_dir": "fibers_transformed",
}

SEGMENT_COLORS = {
    "curve": "blue",
    "connection": "red",
    "approach": "green",
    "leave": "orange",
}


class SlicerApp(ttk.Frame):
    """Main application frame."""

    def __init__(self, master=None):
        super().__init__(master)
        self.master.title("Non-planar slicer — 6-DOF robotic arm")
        self.master.geometry("980x860")

        # The frame itself must be packed inside the root window,
        # otherwise the window shows up empty (all white).
        self.pack(fill=BOTH, expand=True)

        self.vars: dict[str, object] = {}
        self._build_variables()
        self._build_widgets()

        self._result_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._computed: dict | None = None

        self.after(100, self._poll_result_queue)

    # ==================================================================
    # Tk variables
    # ==================================================================

    def _build_variables(self) -> None:
        from tkinter import BooleanVar, DoubleVar, IntVar, StringVar

        v = self.vars

        # ------------------------------------------------------------------
        # File/directory selection
        # ------------------------------------------------------------------
        v["dat_path"] = StringVar(value=str(Path.home()))
        v["fiber_dir"] = StringVar(value=str(Path.home() / DEFAULTS["fiber_dir"]))
        v["transformed_fiber_dir"] = StringVar(
            value=str(Path.home() / DEFAULTS["transformed_fiber_dir"])
        )

        # ------------------------------------------------------------------
        # Printing parameters
        # ------------------------------------------------------------------
        v["layer_height"] = DoubleVar(value=0.35)
        v["layer_width"] = DoubleVar(value=0.70)  # nozzle diameter
        v["n_layers_under_piece"] = IntVar(value=1)

        # ------------------------------------------------------------------
        # Options
        # ------------------------------------------------------------------
        v["rotate_and_lift"] = BooleanVar(value=False)
        v["mold_generation"] = BooleanVar(value=False)

        # ------------------------------------------------------------------
        # Piece path parameters. The contour is defined by a number of
        # contour passes: contour_width = n_layers_contour * layer_width,
        # and the contour isovalues are layer_width * (k + 1/2) for
        # k = 0 .. n_layers_contour - 1.
        # ------------------------------------------------------------------
        v["n_layers_contour"] = IntVar(value=5)
        v["infill_angle"] = StringVar(value="45")
        v["mean_direction_mode"] = StringVar(value="fibers")
        v["approach_distance"] = DoubleVar(value=1.0)
        v["max_connection_length"] = DoubleVar(value=2.0)
        v["merge_tolerance"] = DoubleVar(value=0.1)
        # RDP downsampling of the piece path, applied once before the
        # export AND the visualization so both see the same curves.
        v["simplify_tolerance"] = DoubleVar(value=0.5)

        # ------------------------------------------------------------------
        # Mold parameters
        # ------------------------------------------------------------------
        v["infill_percentage"] = DoubleVar(value=5.0)
        v["n_contour_shells"] = IntVar(value=2)
        v["mold_method"] = StringVar(value="vertical with occlusion")

        # ------------------------------------------------------------------
        # Edge (feature edge) parameters used for the contour computation
        # ------------------------------------------------------------------
        v["feature_angle"] = DoubleVar(value=45.0)
        v["non_manifold_edges"] = BooleanVar(value=True)
        v["boundary_edges"] = BooleanVar(value=True)

        # ------------------------------------------------------------------
        # Visualization options
        # ------------------------------------------------------------------
        v["vis_model"] = BooleanVar(value=True)
        v["vis_edges"] = BooleanVar(value=False)
        v["vis_path"] = BooleanVar(value=True)
        v["vis_legend"] = BooleanVar(value=True)
        v["vis_mold"] = BooleanVar(value=False)
        v["vis_mold_layers"] = BooleanVar(value=False)
        v["vis_mold_layers_range"] = StringVar(value="0")
        v["vis_layers_range"] = StringVar(value="0-4")

        # Export options
        v["export_piece"] = BooleanVar(value=True)
        v["export_piece_dir"] = StringVar(value=str(Path.home() / "trajectories"))
        v["export_mold"] = BooleanVar(value=True)

        # ------------------------------------------------------------------
        # Material parameters
        # ------------------------------------------------------------------
        v["E1"] = DoubleVar(value=DEFAULTS["E1"])
        v["E2"] = DoubleVar(value=DEFAULTS["E2"])
        v["nu12"] = DoubleVar(value=DEFAULTS["nu12"])
        v["G12"] = DoubleVar(value=DEFAULTS["G12"])
        v["G23"] = DoubleVar(value=DEFAULTS["G23"])
        v["E_matrix"] = DoubleVar(value=DEFAULTS["E_matrix"])
        v["nu_matrix"] = DoubleVar(value=DEFAULTS["nu_matrix"])

    # ==================================================================
    # Widget layout
    # ==================================================================

    def _build_widgets(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill=BOTH, expand=True, padx=6, pady=6)

        self.tab_files = ttk.Frame(nb, padding=8)
        self.tab_params = ttk.Frame(nb, padding=8)
        self.tab_mold = ttk.Frame(nb, padding=8)
        self.tab_viz = ttk.Frame(nb, padding=8)
        self.tab_material = ttk.Frame(nb, padding=8)

        nb.add(self.tab_files, text="Files")

        # OAMC fiber path tuning, in a dedicated page (second position)
        from slicer.gui.fibers import FiberTuningTab
        self.tab_fibers = FiberTuningTab(nb, self)
        nb.add(self.tab_fibers, text="Fiber tuning")

        nb.add(self.tab_params, text="Printing parameters")
        nb.add(self.tab_mold, text="Mold")
        nb.add(self.tab_viz, text="Visualization")
        nb.add(self.tab_material, text="Materials")

        self._build_files_tab()
        self._build_params_tab()
        self._build_mold_tab()
        self._build_viz_tab()
        self._build_material_tab()

        # ------------------------------------------------------------------
        # Bottom bar: run button + console-like status
        # ------------------------------------------------------------------
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(fill=X, side=BOTTOM)

        self.run_button = ttk.Button(bar, text="Run slicing", command=self.on_run)
        self.run_button.pack(side=LEFT)

        ttk.Button(bar, text="Quit", command=self.master.destroy).pack(side=RIGHT)

        self.progress = ttk.Progressbar(bar, mode="indeterminate")
        self.progress.pack(side=LEFT, padx=10, fill=X, expand=True)

        self.status = ttk.Label(bar, text="Ready.")
        self.status.pack(side=LEFT, padx=10)

        self.log = ttk.Treeview(self, height=8, columns=("msg",), show="")
        self.log.column("msg", width=1200)
        self.log.pack(fill=BOTH, side=BOTTOM, padx=8, pady=(0, 6))

    # ------------------------------------------------------------------
    def _build_files_tab(self) -> None:
        tab = self.tab_files
        v = self.vars

        def browse_file(var, title, ftypes):
            path = filedialog.askopenfilename(title=title, filetypes=ftypes)
            if path:
                var.set(path)

        def browse_dir(var, title):
            path = filedialog.askdirectory(title=title)
            if path:
                var.set(path)

        f = ttk.LabelFrame(tab, text="ANSYS model (.dat)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text=".dat file:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=v["dat_path"]).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            f, text="Browse…",
            command=lambda: browse_file(
                v["dat_path"], "Select the .dat file", [("ANSYS dat", "*.dat"), ("All files", "*.*")]
            ),
        ).grid(row=0, column=2, padx=4)

        f = ttk.LabelFrame(tab, text="Fibers (CSV files exported by OAMC)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Fiber directory (contains layer_*):").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Entry(f, textvariable=v["fiber_dir"]).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            f, text="Browse…", command=lambda: browse_dir(v["fiber_dir"], "Fiber directory")
        ).grid(row=0, column=2, padx=4)

        ttk.Checkbutton(
            f,
            text="Save transformed fibers (after rotation + lift)",
            variable=v["mold_generation"],
            command=self._on_mold_toggle,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=2)

        ttk.Label(f, text="Transformed fibers directory:").grid(
            row=2, column=0, sticky="w", padx=4, pady=2
        )
        self.entry_transformed_fibers = ttk.Entry(f, textvariable=v["transformed_fiber_dir"])
        self.entry_transformed_fibers.grid(row=2, column=1, sticky="ew", padx=4)
        self.btn_transformed_fibers = ttk.Button(
            f, text="Browse…",
            command=lambda: browse_dir(
                v["transformed_fiber_dir"], "Transformed fibers directory"
            ),
        )
        self.btn_transformed_fibers.grid(row=2, column=2, padx=4)

        f = ttk.LabelFrame(tab, text="Trajectory export")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            f, text="Export the piece trajectory", variable=v["export_piece"]
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4)
        ttk.Label(f, text="Export directory:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=v["export_piece_dir"]).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(
            f, text="Browse…",
            command=lambda: browse_dir(v["export_piece_dir"], "Trajectory export directory"),
        ).grid(row=1, column=2, padx=4)

        self._on_mold_toggle()

    # ------------------------------------------------------------------
    def _build_params_tab(self) -> None:
        tab = self.tab_params
        v = self.vars

        f = ttk.LabelFrame(tab, text="Printing parameters")
        f.pack(fill=X, pady=4)
        for c in range(4):
            f.columnconfigure(c, weight=1)

        rows = [
            ("layer_height", "layer_height [mm] (layer thickness)"),
            ("layer_width", "layer_width [mm] (nozzle diameter)"),
            ("n_layers_under_piece", "n_layers_under_piece (layers under the piece)"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(f, text=label).grid(
                row=i, column=0, columnspan=2, sticky="w", padx=4, pady=2
            )
            ttk.Spinbox(
                f, from_=0.01, to=100.0, increment=0.01, textvariable=v[key], width=10
            ).grid(row=i, column=2, sticky="e", padx=4)

        f = ttk.LabelFrame(tab, text="Model preparation options")
        f.pack(fill=X, pady=4)
        self.chk_rotate = ttk.Checkbutton(
            f,
            text="Rotate and lift model (rotate to align the layer normal with Z)",
            variable=v["rotate_and_lift"],
        )
        self.chk_rotate.pack(anchor="w", padx=4, pady=2)
        self.chk_mold = ttk.Checkbutton(
            f,
            text="Mold generation (generate the mold under the piece — enables rotate and lift)",
            variable=v["mold_generation"],
            command=self._on_mold_toggle,
        )
        self.chk_mold.pack(anchor="w", padx=4, pady=2)

        f = ttk.LabelFrame(
            tab, text="Piece path parameters (contours, connections)"
        )
        f.pack(fill=X, pady=4)
        for c in range(4):
            f.columnconfigure(c, weight=1)

        rows = [
            ("n_layers_contour", "n_layers_contour (number of contour passes)"),
            ("approach_distance", "approach_distance [mm]"),
            ("max_connection_length", "max_connection_length [mm]"),
            ("simplify_tolerance", "simplify_tolerance [mm] (RDP downsampling of the piece path)"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(f, text=label).grid(
                row=i, column=0, columnspan=2, sticky="w", padx=4, pady=2
            )
            ttk.Spinbox(
                f, from_=0.0, to=1000.0, increment=0.1, textvariable=v[key], width=10
            ).grid(row=i, column=2, sticky="e", padx=4)

        ttk.Label(
            f,
            text="The contour width is n_layers_contour × layer_width; the contour "
                 "isovalues are layer_width × (k + 1/2), k = 0 … n_layers_contour − 1.",
            wraplength=760,
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=2)

        f2 = ttk.LabelFrame(tab, text="Infill orientation")
        f2.pack(fill=X, pady=4)
        f2.columnconfigure(1, weight=1)

        ttk.Label(f2, text="Infill angle [°] (relative to the mean direction):").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Combobox(
            f2, textvariable=v["infill_angle"], values=("0", "45", "90"),
            state="readonly", width=6,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(f2, text="Mean direction from:").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Combobox(
            f2,
            textvariable=v["mean_direction_mode"],
            values=("fibers", "dimensions"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(
            f2,
            text="fibers = principal direction of the fiber paths; dimensions = "
                 "longest in-plane axis of the piece (from its dimensions).",
            wraplength=760,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        f = ttk.LabelFrame(
            tab,
            text="Edge extraction (extract_feature_edges — surface boundary "
                 "used for the contour computation)",
        )
        f.pack(fill=X, pady=4)
        for c in range(4):
            f.columnconfigure(c, weight=1)

        rows = [
            ("feature_angle", "feature_angle [°] (angle between adjacent faces)"),
            ("merge_tolerance", "merge_tolerance [mm] (coincident point merging)"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(f, text=label).grid(
                row=i, column=0, columnspan=2, sticky="w", padx=4, pady=2
            )
            ttk.Spinbox(
                f, from_=0.0, to=180.0, increment=0.05, textvariable=v[key], width=10
            ).grid(row=i, column=2, sticky="e", padx=4)
        ttk.Checkbutton(
            f, text="boundary_edges (open borders)", variable=v["boundary_edges"]
        ).grid(row=1, column=0, sticky="w", padx=4)
        ttk.Checkbutton(
            f, text="non_manifold_edges (non-manifold edges)", variable=v["non_manifold_edges"]
        ).grid(row=1, column=1, sticky="w", padx=4)

        info = ttk.Label(
            tab,
            text="Note: contour extraction relies on the surface edges detected "
                 "with these parameters (angle, open borders, non-manifold "
                 "edges). The mold lift guarantees zmin(model) > 0, then adds "
                 "n_layers_under_piece × layer_height.",
            wraplength=900,
        )
        info.pack(anchor="w", padx=4, pady=6)

    # ------------------------------------------------------------------
    def _build_mold_tab(self) -> None:
        tab = self.tab_mold
        v = self.vars

        f = ttk.LabelFrame(tab, text="Mold generation (bottom mold)")
        f.pack(fill=X, pady=4)

        ttk.Label(f, text="Shadow volume method:").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        cb = ttk.Combobox(
            f,
            textvariable=v["mold_method"],
            values=("vertical with occlusion", "cone", "simple"),
            state="readonly",
            width=12,
        )
        cb.grid(row=0, column=1, sticky="w")

        rows = [
            ("infill_percentage", "infill_percentage [%]"),
            ("n_contour_shells", "n_contour_shells (number of mold contour passes)"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=i + 1, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(
                f, from_=0.0, to=100.0, increment=1.0, textvariable=v[key], width=10
            ).grid(row=i + 1, column=1, sticky="w")

        info = ttk.Label(
            tab,
            text="The mold is generated under the piece: the model is first "
                 "lifted so that its zmin is positive, then lifted again by "
                 "n_layers_under_piece × layer_height (see Printing "
                 "parameters). When mold generation is enabled, the model "
                 "lift is enforced.",
            wraplength=900,
        )
        info.pack(anchor="w", padx=4, pady=6)

    # ------------------------------------------------------------------
    def _build_viz_tab(self) -> None:
        tab = self.tab_viz
        v = self.vars

        f = ttk.LabelFrame(
            tab, text="Visualization (external PyVista figure)"
        )
        f.pack(fill=X, pady=4)

        rows = [
            ("vis_model", "Model (piece layer surfaces) — no path computation needed"),
            ("vis_edges", "Detected edges (extract_feature_edges, contour computation) — no path computation needed"),
            ("vis_path", "Piece printing path (curves, connections, approach/leave)"),
            ("vis_mold", "Mold slicing planes — no path computation needed"),
            ("vis_mold_layers", "Mold printing paths (layers of the mold)"),
            ("vis_legend", "Legend"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Checkbutton(f, text=label, variable=v[key]).grid(
                row=i, column=0, sticky="w", padx=4, pady=2
            )

        # The mold layer selection is only relevant when the mold printing
        # paths are actually visualized: keep it disabled otherwise.
        def _on_mold_layers_toggle(*_):
            state = "normal" if v["vis_mold_layers"].get() else "disabled"
            self.entry_mold_layers.configure(state=state)
            self.lbl_mold_layers.configure(foreground="black" if state == "normal" else "gray")

        v["vis_mold_layers"].trace_add("write", _on_mold_layers_toggle)

        ttk.Label(f, text="Piece layers to visualize (e.g. 0-4 or 2,5,7):").grid(
            row=0, column=1, sticky="w", padx=12, pady=2
        )
        ttk.Entry(f, textvariable=v["vis_layers_range"], width=15).grid(
            row=0, column=2, sticky="w"
        )
        self.lbl_mold_layers = ttk.Label(f, text="Mold layers to visualize:")
        self.lbl_mold_layers.grid(row=1, column=1, sticky="w", padx=12, pady=2)
        self.entry_mold_layers = ttk.Entry(
            f, textvariable=v["vis_mold_layers_range"], width=15
        )
        self.entry_mold_layers.grid(row=1, column=2, sticky="w")
        _on_mold_layers_toggle()
        ttk.Label(
            f,
            text="Warning: visualizing several mold printing layers at once can "
                 "crash PyVista — keep the selection small.",
            foreground="red",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=12, pady=2)

        ttk.Label(
            f,
            text="The printing path is drawn with: curves = blue, connections "
                 "= red, approach = green, leave = orange.",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=2)

    # ------------------------------------------------------------------
    def _build_material_tab(self) -> None:
        tab = self.tab_material
        v = self.vars

        f = ttk.LabelFrame(tab, text="Fiber (transversely isotropic)")
        f.pack(fill=X, pady=4)
        for c in range(2):
            f.columnconfigure(c, weight=1)
        rows = [
            ("E1", "E1 [MPa]"),
            ("E2", "E2 [MPa]"),
            ("nu12", "nu12"),
            ("G12", "G12 [MPa]"),
            ("G23", "G23 [MPa]"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(
                f, from_=0.0, to=1e7, increment=10.0, textvariable=v[key], width=12
            ).grid(row=i, column=1, sticky="w")

        f = ttk.LabelFrame(tab, text="Matrix (isotropic)")
        f.pack(fill=X, pady=4)
        for c in range(2):
            f.columnconfigure(c, weight=1)
        rows = [
            ("E_matrix", "E [MPa]"),
            ("nu_matrix", "nu"),
        ]
        for i, (key, label) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(
                f, from_=0.0, to=1e7, increment=10.0, textvariable=v[key], width=12
            ).grid(row=i, column=1, sticky="w")

    # ==================================================================
    # Interaction
    # ==================================================================

    def _on_mold_toggle(self) -> None:
        """Mold generation implies rotate-and-lift; enable related widgets."""
        mold_on = bool(self.vars["mold_generation"].get())
        if mold_on:
            self.vars["rotate_and_lift"].set(True)
        if hasattr(self, "chk_rotate"):
            state = "disabled" if mold_on else "normal"
            self.chk_rotate.configure(state=state)
            self.entry_transformed_fibers.configure(state=state)
            self.btn_transformed_fibers.configure(state=state)

    # ==================================================================
    # Helpers
    # ==================================================================

    def log_message(self, msg: str) -> None:
        self.log.insert("", END, values=(msg,))
        self.log.see(self.log.get_children()[-1])

    def _parse_layer_selection(self, text: str, n_layers: int) -> list[int]:
        """Parse '0-4' or '2,5,7' into a list of valid layer indices."""
        text = text.strip()
        if not text:
            return list(range(min(n_layers, 5)))
        selected = set()
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                selected.update(range(lo_i, hi_i + 1))
            else:
                selected.add(int(part))
        return sorted(i for i in selected if 0 <= i < n_layers)

    def _parse_isovalues(self, text: str) -> list[float]:
        return [float(s) for s in text.replace(";", ",").split(",") if s.strip()]

    # ==================================================================
    # Run
    # ==================================================================

    def on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("Busy", "A computation is already running.")
            return

        self.run_button.configure(state="disabled")
        self.progress.start(20)
        self.status.configure(text="Computing…")
        self.log.delete(*self.log.get_children())

        params = {k: v.get() for k, v in self.vars.items()}
        self._worker = threading.Thread(
            target=self._run_slicing, args=(params, self._result_queue), daemon=True
        )
        self._worker.start()

    @staticmethod
    def _run_slicing(params: dict, result_queue: queue.Queue) -> None:
        """Worker thread: build the model, slice it, export and visualize."""
        try:
            result = SlicerApp._do_slicing(params)
            result_queue.put(("done", result))
        except Exception:
            result_queue.put(("error", traceback.format_exc()))

    # ------------------------------------------------------------------
    @staticmethod
    def _transform_points(points, R, offset) -> np.ndarray:
        """Apply the same rotation and lift as the model to a point array."""
        pts = np.asarray(points, dtype=float)
        if R is not None:
            pts = (R @ pts.T).T
        if offset:
            pts = pts.copy()
            pts[:, 2] += offset
        return pts

    # ------------------------------------------------------------------
    @staticmethod
    def _do_slicing(params: dict) -> dict:
        from oamc.core import CompositeMaterial, CompositeModel
        from oamc.enums import AngleUnit
        from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
        from oamc.integrations.ansys.parser import APDLParser

        dat_path = Path(params["dat_path"])
        if not dat_path.is_file():
            raise FileNotFoundError(f".dat file not found: {dat_path}")

        fiber_dir = Path(params["fiber_dir"])

        layer_height = float(params["layer_height"])
        layer_width = float(params["layer_width"])  # nozzle diameter

        rotate_and_lift = bool(params["rotate_and_lift"])
        mold_generation = bool(params["mold_generation"])
        n_layers_under_piece = int(params["n_layers_under_piece"])

        # ------------------------------------------------------------------
        # 1. Load the model
        # ------------------------------------------------------------------
        parser = APDLParser(dat_path)
        model = parser.get_solid_model()[0]
        mold = parser.get_surface_mesh("MOLD")[0]

        composite_material = CompositeMaterial(
            matrix_material=IsotropicMaterial(
                E=params["E_matrix"], nu=params["nu_matrix"], rho=1.27e-9
            ),
            fiber_material=TransverselyIsotropicMaterial(
                E1=params["E1"], E2=params["E2"], nu12=params["nu12"],
                G12=params["G12"], G23=params["G23"], rho=1.44e-9,
            ),
        )

        model = CompositeModel(
            mesh=model.mesh,
            mold=mold,
            material=composite_material,
            dbc=model.dbc,
            nbc=model.nbc,
            fiber_diameter=layer_width,
            layer_height=layer_height,
        )

        # ------------------------------------------------------------------
        # 2. Rotate and lift
        # ------------------------------------------------------------------
        transformed_dir = Path(params["transformed_fiber_dir"])
        R = None
        lift_offset = 0.0
        if rotate_and_lift or mold_generation:
            R = SlicedPiece.get_rotation_matrix(model)
            SlicedMold.rotate_model(model, R)

            # Lift so that the model's zmin is positive, then add
            # n_layers_under_piece * layer_height below the piece.
            zmin = float(np.min(model.mesh.nodes[:, 2]))
            lift_offset = (n_layers_under_piece * layer_height) - zmin
            SlicedMold.lift_model(model, offset=lift_offset)

        # Save the transformed fibers (rotated + lifted) when the mold
        # generation is active: the GUI then reloads them from this new
        # directory for the slicing.
        fiber_source_dir = fiber_dir
        if mold_generation:
            fiber_paths = SlicedPiece.load_fiber_paths(fiber_dir, layer=1)
            transformed_dir.mkdir(parents=True, exist_ok=True)
            SlicedPiece.save_fiber_paths(
                {k: SlicerApp._transform_points(p, R, lift_offset)
                 for k, p in fiber_paths.items()},
                transformed_dir,
                layer=1,
            )
            fiber_source_dir = transformed_dir

        # ------------------------------------------------------------------
        # 3. Slice the piece
        # ------------------------------------------------------------------
        model.init_q()
        fiber_paths = SlicedPiece.load_fiber_paths(fiber_source_dir, layer=1)
        pieces = SlicedPiece.slice_model(model, fiber_paths=fiber_paths)

        # Merge coincident points of every layer surface right after the
        # isosurface extraction, so that every downstream operation (edge
        # detection, dist_to_boundary, contour extraction, path building,
        # visualization) sees the same merged geometry. build_printing_path
        # re-merges internally (harmless no-op after this point), but
        # previously nothing was merged when the path computation was
        # skipped, which left duplicated boundary points and broke edge
        # detection.
        merge_tolerance = float(params["merge_tolerance"])
        for piece in pieces:
            piece.surface = piece.surface.merge_points(tolerance=merge_tolerance)

        # ------------------------------------------------------------------
        # 4. Mold generation
        # ------------------------------------------------------------------
        mold_layers = None
        shadow_volume = None
        if mold_generation:
            part_surface = SlicedMold._outer_surface(model)
            method = params["mold_method"]

            if method == "simple":
                shadow_volume = SlicedMold.get_shadow_volume(model, plane_z=0.0)
            elif method == "cone":
                shadow_volume = SlicedMold.get_shadow_volume_cone(model, plane_z=0.0)
            else:  # vertical with occlusion (default)
                shadow_volume, _, _ = SlicedMold.get_shadow_volume_vertical(
                    model, plane_z=0.0
                )

            mold_layers = SlicedMold.slice_shadow_volume(
                shadow_volume=shadow_volume,
                part_surface=part_surface,
                layer_height=layer_height,
                nozzle_diameter=layer_width,
                n_contour_shells=int(params["n_contour_shells"]),
                infill_percentage=float(params["infill_percentage"]),
                plane_z=0.0,
            )

        # ------------------------------------------------------------------
        # 5. Piece printing path (only when needed: visualization of the
        #    path, or trajectory export)
        # ------------------------------------------------------------------
        n_layers_contour = int(params["n_layers_contour"])
        contour_width = n_layers_contour * layer_width
        contour_isovalues = [
            layer_width * (k + 0.5) for k in range(n_layers_contour)
        ]

        # Infill orientation: angle to the mean direction (0/45/90) and how
        # the mean direction itself is obtained.
        infill_angle = float(params["infill_angle"])
        mean_direction_mode = str(params["mean_direction_mode"])
        for piece in pieces:
            piece.infill_angle_deg = infill_angle
            piece.mean_direction_mode = mean_direction_mode

        vis_path = bool(params["vis_path"])
        export_piece = bool(params["export_piece"])

        piece_paths = []
        if vis_path or export_piece:
            feature_edge_params = {
                "feature_angle": float(params["feature_angle"]),
                "boundary_edges": bool(params["boundary_edges"]),
                "non_manifold_edges": bool(params["non_manifold_edges"]),
            }

            for i, piece in enumerate(pieces):
                piece.feature_edge_params = feature_edge_params
                segments, Lcmd, types = piece.build_printing_path(
                    i_surface=i,
                    contour_width=contour_width,
                    approach_distance=float(params["approach_distance"]),
                    max_connection_length=float(params["max_connection_length"]),
                    contour_isovalues=contour_isovalues,
                    merge_tolerance=float(params["merge_tolerance"]),
                    layer_width=layer_width,
                )

                # Downsample once, here: the export and the visualization
                # both reuse these already-simplified curves.
                segments, types = piece.analyze_and_downsample_path(
                    segments,
                    types,
                    tolerance=float(params["simplify_tolerance"]),
                    min_segment_length=2.0 * layer_width,
                )
                piece_paths.append((segments, types))

        # ------------------------------------------------------------------
        # 6. Export
        # ------------------------------------------------------------------
        export_dir = Path(params["export_piece_dir"])
        if export_piece:
            export_dir.mkdir(parents=True, exist_ok=True)
            SlicedPiece.export_layers_to_txt(
                pieces,
                export_dir / "piece_path.txt",
                piece_paths=piece_paths,
                include_normal=True,
            )

        if mold_generation and mold_layers and params["export_mold"]:
            SlicedMold.export_layers_to_txt(
                mold_layers,
                export_dir / "mold_path.txt",
                nozzle_diameter=layer_width,
                infill_percentage=float(params["infill_percentage"]),
                approach_distance=float(params["approach_distance"]),
                layer_height=layer_height,
            )

        return {
            "model": model,
            "pieces": pieces,
            "piece_paths": piece_paths,
            "mold_layers": mold_layers,
            "shadow_volume": shadow_volume,
            "params": params,
            "layer_width": layer_width,
            "layer_height": layer_height,
        }

    # ------------------------------------------------------------------
    # Result handling (runs in the GUI thread)
    # ------------------------------------------------------------------

    def _poll_result_queue(self) -> None:
        try:
            kind, payload = self._result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result_queue)
            return

        self.progress.stop()
        self.run_button.configure(state="normal")

        if kind == "error":
            self.status.configure(text="Error during the computation.")
            self.log_message("ERROR:")
            for line in payload.strip().splitlines()[-30:]:
                self.log_message(line)
            messagebox.showerror("Error", payload.strip().splitlines()[-1])
        else:
            self.status.configure(text="Computation finished.")
            self.log_message("Computation finished.")
            self._computed = payload
            try:
                self._show_visualization(payload)
            except Exception:
                err = traceback.format_exc()
                self.log_message("ERROR (visualization):")
                for line in err.strip().splitlines()[-20:]:
                    self.log_message(line)
                messagebox.showerror("Error (visualization)", err.strip().splitlines()[-1])

        self.after(100, self._poll_result_queue)

    # ==================================================================
    # Visualization
    # ==================================================================

    def _show_visualization(self, result: dict) -> None:
        if pv is None:
            raise ImportError("pyvista is required for the visualization")

        params = result["params"]
        pieces = result["pieces"]
        piece_paths = result["piece_paths"]
        mold_layers = result["mold_layers"]
        shadow_volume = result["shadow_volume"]

        vis_layers = self._parse_layer_selection(
            str(params["vis_layers_range"]), len(pieces)
        )

        plotter = pv.Plotter(title="Slicer — visualization")

        # ------------------------------------------------------------------
        # Model (layer surfaces) + edges
        # ------------------------------------------------------------------
        if params["vis_model"]:
            for i in vis_layers:
                if 0 <= i < len(pieces):
                    plotter.add_mesh(
                        pieces[i].surface,
                        color="lightblue",
                        opacity=0.1,
                        show_edges=True,
                        label="Model (layer)" if i == vis_layers[0] else None,
                    )

        if params["vis_edges"]:
            for i in vis_layers:
                if 0 <= i < len(pieces):
                    edges = pieces[i].surface.extract_feature_edges(
                        feature_angle=params["feature_angle"],
                        boundary_edges=params["boundary_edges"],
                        non_manifold_edges=params["non_manifold_edges"],
                    )
                    if edges.n_points:
                        plotter.add_mesh(
                            edges,
                            color="black",
                            line_width=3,
                            label="Detected edges" if i == vis_layers[0] else None,
                        )

        # ------------------------------------------------------------------
        # Printing path (curves / connections / approach / leave)
        # ------------------------------------------------------------------
        if params["vis_path"]:
            for i in vis_layers:
                if 0 <= i < len(piece_paths):
                    segments, types = piece_paths[i]
                    for seg, seg_type in zip(segments, types):
                        first = i == vis_layers[0]
                        plotter.add_mesh(
                            pv.lines_from_points(np.asarray(seg), close=False),
                            color=SEGMENT_COLORS.get(seg_type, "gray"),
                            line_width=4 if seg_type == "connection" else 2,
                            label=seg_type if first else None,
                        )

        # ------------------------------------------------------------------
        # Mold: plane-based visualization (dedicated branch implementation)
        # ------------------------------------------------------------------
        if params["vis_mold"] and shadow_volume is not None:
            patches_x, patches_y = build_slicing_plane_patches(
                shadow_volume,
                nozzle_diameter=float(params["layer_width"]),
                infill_percentage=float(params["infill_percentage"]),
            )
            for j, patch in enumerate(patches_x):
                plotter.add_mesh(
                    patch,
                    color="blue",
                    opacity=0.35,
                    label="Mold — planes ⊥ X" if j == 0 else None,
                )
            for j, patch in enumerate(patches_y):
                plotter.add_mesh(
                    patch,
                    color="red",
                    opacity=0.35,
                    label="Mold — planes ⊥ Y" if j == 0 else None,
                )
        elif params["vis_mold"] and mold_layers:
            # No shadow volume (should not happen): fall back on the layers
            for j, layer in enumerate(mold_layers[:2]):
                for poly in layer.contour_shells:
                    plotter.add_mesh(
                        pv.lines_from_points(poly, close=False),
                        color="orange",
                        line_width=2,
                        label="Mold (contours)" if j == 0 else None,
                    )

        # ------------------------------------------------------------------
        # Mold printing paths (per layer). Visualizing several layers at
        # once can crash PyVista, so the selection is meant to stay small.
        # ------------------------------------------------------------------
        if params["vis_mold_layers"] and mold_layers:
            vis_mold_layers = self._parse_layer_selection(
                str(params["vis_mold_layers_range"]), len(mold_layers)
            )
            for j, idx in enumerate(vis_mold_layers):
                layer = mold_layers[idx]
                segments, types = layer.build_support_path(
                    nozzle_diameter=float(params["layer_width"]),
                    infill_percentage=float(params["infill_percentage"]),
                    approach_distance=float(params["approach_distance"]),
                    layer_height=float(params["layer_height"]),
                )
                first = j == 0
                for seg, seg_type in zip(segments, types):
                    plotter.add_mesh(
                        pv.lines_from_points(np.asarray(seg), close=False),
                        color=SEGMENT_COLORS.get(seg_type, "gray"),
                        line_width=4 if seg_type == "connection" else 2,
                        label=f"Mold path ({seg_type})" if first else None,
                    )

        if params["vis_legend"]:
            plotter.add_legend()

        plotter.show()


def run_gui() -> None:
    """Entry point: create the Tk root window and start the app."""
    import tkinter as tk

    root = tk.Tk()
    SlicerApp(master=root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
