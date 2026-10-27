"""Fiber path tuning tab (OAMC pipeline) for the slicer GUI.

Reproduces the functional pipeline of the standalone OAMC fiber tuner
(bearing load, mold selection, least-squares fiber optimization, load
paths, post-processing and fiber export) inside a dedicated tab of the
slicer GUI. Only the functions are reused, not the original layout.
"""

from __future__ import annotations

import csv
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, X, filedialog, messagebox, ttk

import numpy as np

try:
    from oamc.core import CompositeMaterial, CompositeModel
    from oamc.enums import AngleUnit, Direction, ProjectionMethod
    from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
    from oamc.integrations.ansys.parser import APDLParser
    from oamc.logging import enable_logging
    from oamc.lpp import LPP
    from oamc.post import Viewer

    OAMC_OK = True
except ImportError:  # pragma: no cover
    OAMC_OK = False


class FiberTuningTab(ttk.Frame):
    """Tab embedding the OAMC fiber path optimization pipeline."""

    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app

        self.vars: dict[str, object] = {}
        self._build_variables()
        self._build_widgets()

        self._running = False
        self._model = None
        self._viewer = None
        self._lpp = None
        self._data: list[dict] = []

    # ==================================================================
    # Variables
    # ==================================================================

    def _build_variables(self) -> None:
        v = self.vars

        # Pipeline / example selection
        v["example"] = StringVar(value="fdm")  # "lpp" | "fdm" | "fdm_csv"

        # Output
        v["out_dir"] = StringVar(value=str(Path.home() / "oamc_output"))

        # Bearing load
        v["apply_load"] = BooleanVar(value=True)
        v["hole_ns"] = StringVar(value="HOLE")
        v["fx"] = DoubleVar(value=0.0)
        v["fy"] = DoubleVar(value=-1000.0)
        v["fz"] = DoubleVar(value=0.0)

        # Mold
        v["use_mold"] = BooleanVar(value=True)
        v["mold_ns"] = StringVar(value="MOLD")

        # Optimization
        v["n_iter"] = IntVar(value=5)
        v["v_min"] = DoubleVar(value=0.2)
        v["v_max"] = DoubleVar(value=1.0)
        v["p_splits"] = IntVar(value=2)
        v["min_len"] = DoubleVar(value=40.0)

        # Post-processing
        v["rdp_dev"] = DoubleVar(value=0.05)
        v["out_maxl"] = DoubleVar(value=10.0)
        v["out_ang"] = DoubleVar(value=45.0)

        # Load paths (LPP)
        v["use_seeds_file"] = BooleanVar(value=True)
        v["seeds_path"] = StringVar(value=str(Path.home() / "seeds.txt"))
        v["seeds_ns"] = StringVar(value="SEEDS")
        v["direction"] = StringVar(value="MAX")
        v["step_size"] = DoubleVar(value=1.0)

        # Visualization
        v["opacity"] = DoubleVar(value=0.3)
        v["show_edges"] = BooleanVar(value=False)
        v["f_scale"] = DoubleVar(value=10.0)
        v["u_scale"] = DoubleVar(value=10.0)

        # Export
        v["angle_unit"] = StringVar(value="DEG360")
        v["convention"] = StringVar(value="unit vector")
        v["subdirs"] = BooleanVar(value=True)

    # ==================================================================
    # Widgets
    # ==================================================================

    def _build_widgets(self) -> None:
        v = self.vars

        # ------------------------------------------------------------------
        # Pipeline selection + run
        # ------------------------------------------------------------------
        f = ttk.LabelFrame(self, text="Fiber pipeline (OAMC)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Pipeline:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(
            f,
            textvariable=v["example"],
            values=("lpp", "fdm", "fdm_csv"),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            f,
            text="lpp = load paths (needs seeds), fdm = FDM fiber paths, "
                 "fdm_csv = FDM + compliance/convergence CSV.",
        ).grid(row=0, column=2, sticky="w", padx=8)

        ttk.Label(f, text="Output directory:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=v["out_dir"]).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(
            f, text="Browse…",
            command=lambda: self._browse_dir(v["out_dir"]),
        ).grid(row=1, column=2, padx=4)

        run_bar = ttk.Frame(f)
        run_bar.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=4)
        self.run_button = ttk.Button(run_bar, text="Run fiber pipeline", command=self.on_run)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(run_bar, mode="indeterminate", length=160)
        self.progress.pack(side="left", padx=8)
        self.status = ttk.Label(run_bar, text="Ready.")
        self.status.pack(side="left", padx=8)

        # ------------------------------------------------------------------
        # Parameters (two columns of LabelFrames)
        # ------------------------------------------------------------------
        cols = ttk.Frame(self)
        cols.pack(fill=X, pady=4)
        left = ttk.Frame(cols)
        right = ttk.Frame(cols)
        left.pack(side="left", fill=X, expand=True, padx=(0, 6))
        right.pack(side="left", fill=X, expand=True)

        # --- Left column ---
        f = ttk.LabelFrame(left, text="Bearing load")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            f, text="Apply bearing load", variable=v["apply_load"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(f, text="Named selection:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=v["hole_ns"]).grid(row=1, column=1, sticky="ew")
        for i, (key, label) in enumerate([
            ("fx", "Fx [N]"), ("fy", "Fy [N]"), ("fz", "Fz [N]"),
        ], start=2):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(f, from_=-1e6, to=1e6, increment=10.0, textvariable=v[key], width=10).grid(
                row=i, column=1, sticky="w"
            )

        f = ttk.LabelFrame(left, text="Mold (surface mesh)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            f, text="Use a mold (named selection)", variable=v["use_mold"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(f, text="Mold NS:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=v["mold_ns"]).grid(row=1, column=1, sticky="ew")

        f = ttk.LabelFrame(left, text="Optimization (least squares)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        rows = [
            ("n_iter", "iterations", 1, 20, 1),
            ("v_min", "v_min", 0.0, 1.0, 0.01),
            ("v_max", "v_max", 0.0, 1.0, 0.01),
            ("p_splits", "p_splits", 1, 6, 1),
            ("min_len", "min_length [mm]", 1, 200, 1),
        ]
        for i, (key, label, lo, hi, inc) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(
                f, from_=lo, to=hi, increment=inc, textvariable=v[key], width=10
            ).grid(row=i, column=1, sticky="w")

        f = ttk.LabelFrame(left, text="Post-processing")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        rows = [
            ("rdp_dev", "RDP max deviation [mm]", 0.001, 1.0, 0.001),
            ("out_maxl", "Outlier max_length [mm]", 0.5, 50.0, 0.5),
            ("out_ang", "Outlier min_angle [°]", 0, 90, 1),
        ]
        for i, (key, label, lo, hi, inc) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Spinbox(
                f, from_=lo, to=hi, increment=inc, textvariable=v[key], width=10
            ).grid(row=i, column=1, sticky="w")

        # --- Right column ---
        f = ttk.LabelFrame(right, text="Load paths (lpp pipeline)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            f, text="Seeds from file", variable=v["use_seeds_file"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(f, text="seeds.txt:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=v["seeds_path"]).grid(row=1, column=1, sticky="ew")
        ttk.Button(
            f, text="Browse…", command=lambda: self._browse_file(v["seeds_path"])
        ).grid(row=1, column=2, padx=4)
        ttk.Label(f, text="Seeds named selection (if no file):").grid(
            row=2, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Entry(f, textvariable=v["seeds_ns"]).grid(row=2, column=1, sticky="ew")
        ttk.Label(f, text="Direction:").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(
            f, textvariable=v["direction"], values=("MAX", "MIN", "MID"),
            state="readonly", width=8,
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(f, text="Step size [mm]:").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(
            f, from_=0.1, to=20.0, increment=0.1, textvariable=v["step_size"], width=10
        ).grid(row=4, column=1, sticky="w")

        f = ttk.LabelFrame(right, text="Visualization (OAMC Viewer)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Opacity:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(
            f, from_=0.0, to=1.0, increment=0.05, textvariable=v["opacity"], width=10
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="f_scaling_factor:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(
            f, from_=0.0, to=100.0, increment=0.5, textvariable=v["f_scale"], width=10
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(f, text="u_scaling_factor:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(
            f, from_=0.0, to=100.0, increment=0.5, textvariable=v["u_scale"], width=10
        ).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(
            f, text="Show edges", variable=v["show_edges"]
        ).grid(row=3, column=0, sticky="w", padx=4)

        f = ttk.LabelFrame(right, text="Fiber export (save_fibers)")
        f.pack(fill=X, pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Angle unit:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(
            f, textvariable=v["angle_unit"], values=("DEG360", "DEG180", "RAD"),
            state="readonly", width=10,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Convention:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(
            f, textvariable=v["convention"], values=("unit vector", "euler angles"),
            state="readonly", width=12,
        ).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(
            f, text="Subdirectories (layer_i folders)", variable=v["subdirs"]
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=4)

        info = ttk.Label(
            self,
            text="The .dat model, materials, fiber diameter (= layer_width) and "
                 "layer_height are shared with the slicer tabs (Files / Materials / "
                 "Printing parameters).",
            wraplength=900,
        )
        info.pack(anchor="w", padx=4, pady=6)

    # ==================================================================
    # Helpers
    # ==================================================================

    def log(self, msg: str) -> None:
        self.app.log_message(msg)

    @staticmethod
    def _browse_file(var) -> None:
        path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if path:
            var.set(path)

    @staticmethod
    def _browse_dir(var) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # ==================================================================
    # Run
    # ==================================================================

    def on_run(self) -> None:
        if self._running:
            messagebox.showwarning("Busy", "A fiber computation is already running.")
            return
        if not OAMC_OK:
            messagebox.showerror(
                "oamc missing", "The 'oamc' package is not installed."
            )
            return

        self._running = True
        self.run_button.configure(state="disabled")
        self.progress.start(15)
        self.status.configure(text="Computing…")

        params = {k: var.get() for k, var in self.vars.items()}
        t = threading.Thread(target=self._run_pipeline, args=(params,), daemon=True)
        t.start()

    def _finish(self, error: str | None) -> None:
        def apply():
            self.progress.stop()
            self.run_button.configure(state="normal")
            if error:
                self.status.configure(text="Error.")
                messagebox.showerror("Error", error)
            else:
                self.status.configure(text="Done.")
        self.after(0, apply)

    # ------------------------------------------------------------------
    def _run_pipeline(self, params: dict) -> None:
        error = None
        try:
            from oamc.core import CompositeMaterial, CompositeModel
            from oamc.enums import AngleUnit, Direction, ProjectionMethod
            from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
            from oamc.integrations.ansys.parser import APDLParser
            from oamc.logging import enable_logging
            from oamc.lpp import LPP
            from oamc.post import Viewer

            # Shared parameters from the slicer tabs
            app_params = {k: var.get() for k, var in self.app.vars.items()}
            dat_path = Path(app_params["dat_path"])
            if not dat_path.is_file():
                raise FileNotFoundError(f".dat file not found: {dat_path}")

            out_path = Path(params["out_dir"])
            out_path.mkdir(parents=True, exist_ok=True)

            enable_logging()

            self.log("Parsing the .dat model…")
            parser = APDLParser(dat_path)

            if params["apply_load"]:
                force = (params["fx"], params["fy"], params["fz"])
                parser.apply_bearing_load(force=force, target=params["hole_ns"])
                self.log(f"Bearing load applied: {force} N -> '{params['hole_ns']}'")
            else:
                self.log("Bearing load disabled (load read from the .dat file).")

            example = params["example"]
            if example == "lpp":
                self._run_load_paths(parser, params, out_path)
            else:
                self._run_fdm(
                    parser, params, out_path, app_params,
                    with_csv=(example == "fdm_csv"),
                )
        except Exception:
            error = traceback.format_exc().strip().splitlines()[-1]
            self.log("ERROR (fiber pipeline):")
            for line in traceback.format_exc().strip().splitlines()[-20:]:
                self.log(line)
        finally:
            self._finish(error)

    # ------------------------------------------------------------------
    def _get_mold(self, parser, params):
        if not params["use_mold"]:
            self.log("Mold disabled — CompositeModel without mold surface.")
            return None
        ns = str(params["mold_ns"]).strip()
        if not ns:
            self.log("Mold NS empty — ignored.")
            return None
        try:
            mold, *_ = parser.get_surface_mesh(ns)
            self.log(f"Mold '{ns}' loaded.")
            return mold
        except (ValueError, KeyError) as exc:
            self.log(f"Mold '{ns}' not found ({exc}) — continuing without mold.")
            return None

    def _build_composite_model(self, parser, params, app_params):
        from oamc.core import CompositeMaterial, CompositeModel
        from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial

        model_raw, *_ = parser.get_solid_model()
        mold = self._get_mold(parser, params)

        material = CompositeMaterial(
            matrix_material=IsotropicMaterial(
                E=app_params["E_matrix"],
                nu=app_params["nu_matrix"],
                rho=1.27e-9,
            ),
            fiber_material=TransverselyIsotropicMaterial(
                E1=app_params["E1"],
                E2=app_params["E2"],
                nu12=app_params["nu12"],
                G12=app_params["G12"],
                G23=app_params["G23"],
                rho=1.44e-9,
            ),
        )

        kwargs = dict(
            mesh=model_raw.mesh,
            material=material,
            dbc=model_raw.dbc,
            nbc=model_raw.nbc,
            fiber_diameter=app_params["layer_width"],
            layer_height=app_params["layer_height"],
        )
        if mold is not None:
            kwargs["mold"] = mold
        return CompositeModel(**kwargs)

    # ------------------------------------------------------------------
    def _view(self, model, params, paths, title):
        from oamc.enums import ProjectionMethod
        from oamc.post import Viewer

        viewer = Viewer(model=model, title=title)
        self._viewer = viewer
        viewer.view(
            show_edges=params["show_edges"],
            show_origin=False,
            f_scaling_factor=params["f_scale"],
            u_scaling_factor=params["u_scale"],
            projection_method=ProjectionMethod.L2,
            opacity=params["opacity"],
            paths=paths,
        )

    # ------------------------------------------------------------------
    def _run_load_paths(self, parser, params, out_path: Path) -> None:
        from oamc.enums import Direction
        from oamc.lpp import LPP

        self.log("Pipeline: load paths (LPP)")
        model, n2i, _ = parser.get_solid_model()

        if params["use_seeds_file"]:
            seeds = Path(params["seeds_path"])
            if not seeds.is_file():
                raise FileNotFoundError(f"seeds file not found: {seeds}")
        else:
            seed_nums = parser.get_named_selection(params["seeds_ns"])
            seed_idx = [n2i[n] for n in seed_nums]
            seeds = model.mesh.nodes[seed_idx]
            self.log(f"Seeds: {len(seeds)} nodes from '{params['seeds_ns']}'")

        dir_map = {"MAX": Direction.MAX, "MIN": Direction.MIN, "MID": Direction.MID}
        direction = dir_map.get(params["direction"], Direction.MAX)

        lpp = LPP(model)
        self._lpp = lpp
        lpp.generate_load_paths(
            direction=direction,
            step_size=params["step_size"],
            seeds=seeds,
        )
        self.log(f"{len(lpp.paths)} load paths generated.")

        lpp.save_load_paths(directory=out_path / "paths", file_format="CSV")
        self.log(f"Load paths saved to {out_path / 'paths'}")

        self._view(model, params, lpp.paths, "OAMC — Load Paths")

    # ------------------------------------------------------------------
    def _run_fdm(self, parser, params, out_path: Path, app_params, with_csv: bool) -> None:
        import csv as _csv

        self.log("Pipeline: FDM fiber paths" + (" + CSV" if with_csv else ""))
        model = self._build_composite_model(parser, params, app_params)
        self._model = model

        C_0 = model.compliance(model.p)
        self.log(f"Initial compliance: {C_0:.3f} mJ")

        model.init_q()

        data = []
        if with_csv:
            data.append({
                "iteration": 0,
                "total fiber length from scalar fields": 0,
                "exact total fiber length": 0,
                "average fiber-filament volume fraction": 0,
                "compliance": round(C_0, 3),
                "fiber efficiency": 0,
                "relative compliance with fiber penalization": 0,
            })

        n_iter = int(params["n_iter"])
        for i in range(n_iter):
            self.log(f"Iteration {i + 1}/{n_iter}…")
            model.init_p_by_least_squares(
                v_min=params["v_min"],
                v_max=params["v_max"],
            )
            model.compute_fibers(
                p_splits=int(params["p_splits"]),
                min_length=params["min_len"],
            )
            L_f = model.precise_total_length
            V_f = L_f * model.fiber_area / model.mesh.volume
            C = model.compliance(model.p)
            self.log(
                f"  C={C:.3f} mJ  L_f={L_f:.1f} mm  V_f={V_f:.3f}  "
                f"eta_f={(1 - C / C_0) / V_f:.3f}"
            )
            if with_csv:
                data.append({
                    "iteration": i + 1,
                    "total fiber length from scalar fields": round(
                        model.total_length(model.p)[0], 3),
                    "exact total fiber length": round(L_f, 3),
                    "average fiber-filament volume fraction": round(V_f, 3),
                    "compliance": round(C, 3),
                    "fiber efficiency": round((1 - C / C_0) / V_f, 3),
                    "relative compliance with fiber penalization": round(
                        (C / C_0) ** 2 + V_f ** 2, 3),
                })

        if with_csv and data:
            csv_path = out_path / "oamc_results.csv"
            with open(csv_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            self.log(f"CSV exported -> {csv_path}")

        self._post_process(model, params, out_path)
        self._view(model, params, model.fibers_as_list, "OAMC — FDM Fiber Paths")

    # ------------------------------------------------------------------
    def _post_process(self, model, params, out_path: Path) -> None:
        from oamc.enums import AngleUnit

        n_before = model.total_number_of_points
        model.downsample_fibers_by_rdp(max_deviation=params["rdp_dev"])
        n_after_rdp = model.total_number_of_points
        model.remove_outliers(
            max_length=params["out_maxl"],
            min_angle=params["out_ang"] * np.pi / 180.0,
        )
        n_after_out = model.total_number_of_points
        self.log(
            f"Points: {n_before} -> {n_after_rdp} (RDP) -> {n_after_out} (outliers)"
        )

        angle_map = {"DEG360": AngleUnit.DEG360, "DEG180": AngleUnit.DEG180, "RAD": AngleUnit.RAD}
        model.save_fibers(
            directory=out_path / "fibers",
            subdirectories=bool(params["subdirs"]),
            convention=str(params["convention"]),
            angle_unit=angle_map.get(params["angle_unit"], AngleUnit.DEG360),
        )
        self.log(f"Fibers saved -> {out_path / 'fibers'}")
