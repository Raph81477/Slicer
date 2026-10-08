"""Load-Based Generation of Fiber Paths for FDM Printing"""

from pathlib import Path

import pyvista as pv

from oamc.constants import BANNER
from oamc.core import CompositeMaterial, CompositeModel
from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging
from oamc.post import Viewer

from SlicedPiece import SlicedPiece
from SlicedMold import SlicedMold

DIR = Path(__file__).parent.resolve()


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
        fiber_diameter=0.70,
        layer_height=0.35,
    )

    R = SlicedPiece.get_rotation_matrix(model)
    print(f"Rotation matrix:\n{R}")
    SlicedMold.rotate_model(model, R)
    SlicedMold.lift_model(model, offset=20)

    viewer = Viewer(
        model=model,
        title="OAMC Example 2: Load-Based Generation of Fiber Paths for FDM Printing",
    )

    # ---------------------------------------------------------------------------------------------

    C_0 = model.compliance(model.p)
    print(f"Structural compliance before initialization: {round(C_0, 3)} mJ")

    model.init_q()

    fiber_paths = SlicedPiece.load_fiber_paths(DIR, layer=1)
    pieces = SlicedPiece.slice_model(model, fiber_paths=fiber_paths)

    #### Plot the lower surfaces

    # Exaggerated spacing between the plotted surfaces, for readability
    # only - the real layer_height (0.35) is far too thin to tell layers
    # apart visually. The printing path itself uses the real coordinates.
    vis_layer_height = 2
    plotter = pv.Plotter()
    n = pieces[0].normal

    colors = {"curve": "blue", "connection": "red", "approach": "green", "leave": "orange"}

    n_layers_to_plot = 2

    print("Total number of layers :", len(pieces))
    for i, piece in enumerate(pieces[0:2]):

        # Offset a copy of the surface along the normal, for visual
        # separation only.
        surface_vis = piece.surface.copy()
        surface_vis.points = piece.surface.points + i * vis_layer_height * n
        surface_vis.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        plotter.add_mesh(
            surface_vis,
            color="lightblue",   # surface color
            opacity=0.1,         # transparency
            show_edges=True,     # show edges
            label="Print surface",
        )

        ###############################################################
        # Build and plot this layer's printing path
        ###############################################################

        path_segments, Lcmd, segment_types = piece.build_printing_path(
            i_surface=i,
            contour_width=3.0,
            approach_distance=1.0,
            max_connection_length=4 * 0.5,
        )

        print(f"Layer {i}: full path = {len(path_segments)} segments, {len(Lcmd)} commands")

        for seg, seg_type in zip(path_segments, segment_types):
            plotter.add_mesh(
                pv.lines_from_points(seg, close=False),
                color=colors[seg_type],
                line_width=4 if seg_type == "connection" else 2,
            )

        # Export the printing path of every layer to a single text file, ready
    # to be loaded by an external trajectory simulator such as RoboDK.
    SlicedPiece.export_layers_to_txt(
        pieces[0:1],
        DIR / "trajectories" / "piece_path.txt",
        contour_width=3.0,
        approach_distance=1.0,
        max_connection_length=4 * 0.5,
        include_orientation=True,
        simplify_tolerance=0.5, #mm
    )
    plotter.show()


if __name__ == "__main__":
    main()
