"""Load-Based Generation of Fiber Paths for FDM Printing"""

from pathlib import Path

import pyvista as pv

from oamc.constants import BANNER
from oamc.core import CompositeMaterial, CompositeModel
from oamc.enums import AngleUnit, ProjectionMethod
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
        layer_height=1.0,
    )

    SlicedMold.rotate_model(model, SlicedPiece.get_rotation_matrix(model))
    SlicedMold.lift_model(model, offset=65)

    viewer = Viewer(
        model=model,
        title="OAMC — Example 5 — C-Beam",
    )

    C_0 = model.compliance(model.p)
    print(f"Structural compliance before initialization: {round(C_0, 3)} mJ")

    model.init_q()

    # for i in range(1):
    #     print(f"- Initialization iteration {i + 1} -")
    #     model.init_p_by_least_squares(v_min=0.1, v_max=0.1)

    #     model.compute_fibers(p_splits=2, min_length=50)

    #     L_f = model.precise_total_length
    #     V_f = L_f * model.fiber_area / model.mesh.volume
    #     print(f"Structural compliance = {round(model.compliance(model.p), 3)} mJ")
    #     print(
    #         f"Total fiber length from scalar fields = {round(model.total_length(model.p)[0], 3)} mm"
    #     )
    #     print(f"Precise total fiber length = {round(L_f, 3)} mm")
    #     print(
    #         f"Total fiber weight = {round(L_f * model.fiber_area * fiber_material.rho * 1e6, 3)} g"
    #     )
    #     print(f"Average fiber volume fraction = {round(V_f, 3)}")
    #     print(
    #         f"eta_f = (1 - C / C_0) / V_f = {round((1 - model.compliance(model.p) / C_0) / V_f, 3)}"
    #     )
    #     print(
    #         f"c_p = (C / C_0)**2 + 1 * V_f**2 = {round((model.compliance(model.p) / C_0) ** 2 + V_f**2, 3)}"
    #     )

    # print("- Gradient-based optimization -")

    # def callback(*args) -> None:
    #     print(args)

    # model.optimize_p(
    #     max_fiber_length=1e4,
    #     callback=callback,
    #     iteration_limit=5,
    # )

    # print(f"Structural compliance after optimization: {round(model.compliance(model.p), 3)} mJ")

    # print("- Field post-processing -")

    # model.filter_p_by_diffusion(iterations=10, diffusion_level=0.01)

    # print(f"Structural compliance after diffusion filtering: {round(model.compliance(model.p), 3)} mJ")

    print("- Polyline post-processing -")

    model.compute_fibers(p_splits=2, min_length=50)

    print(f"Total number of points before downsampling: {model.total_number_of_points}")
    model.downsample_fibers_by_rdp(max_deviation=0.2)
    print(f"Total number of points after downsampling: {model.total_number_of_points}")
    # model.remove_outliers(max_length=10, min_angle=numpy.pi / 4)
    # print(f"Total number of points after removing outliers: {model.total_number_of_points}")

    model.save_fibers(
        directory=DIR / "fibers",
        subdirectories=True,
        convention="unit vector",
        angle_unit=AngleUnit.DEG360,
    )

    base_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=200,   # displayed plane width (adjust to part dimensions)
        j_size=200,   # displayed plane height
    )

    # shadow_volume = SlicedMold.get_shadow_volume(model, plane_z=0.0)
    # shadow_volume = SlicedMold.get_shadow_volume_cone(
    #     model,
    #     plane_z=0.0,
    #     offset_faces=0.1,
    #     cone_angle=45.0,
    #     no_rays=8,
    #     start_phase=0.0,
    #     distance_from_contour=0,
    # )

    shadow_volume, cloud_top, cloud_bot = SlicedMold.get_shadow_volume_vertical(
        model,
        plane_z=0.0,
        offset_faces=0.05,
    )

    # Just to visualize the distribution of top_pts and bottom_pts
    # viewer.plotter.add_mesh(cloud_top, color="red",   point_size=6, render_points_as_spheres=True)
    # viewer.plotter.add_mesh(cloud_bot, color="green", point_size=6, render_points_as_spheres=True)

    # viewer.plotter.add_mesh(shadow_volume, color="red", opacity=0.3, show_edges=True)

    viewer.plotter.add_mesh(base_plane, color="gray", opacity=0.3, show_edges=True)

    pv_mesh = SlicedMold.solid_mesh_to_pyvista(model.mesh)
    linear_mesh = pv_mesh.linear_copy()
    surface = linear_mesh.extract_surface(algorithm='geometry')
    # surface = surface.clean(tolerance=1e-6)
    # surface = surface.triangulate()

    viewer.plotter.add_mesh(surface)

    # # Generate the planes
    # planes_x, planes_y = SlicedMold.generate_support_slicing_planes(
    #     shadow_volume=shadow_volume,
    #     infill_percentage=20.0,
    #     plane_z=0.0,
    # )

    # # Generate the polylines
    # infill_x, infill_y = SlicedMold.generate_support_infill_lines(
    #     shadow_volume=shadow_volume,
    #     planes_x=planes_x,
    #     planes_y=planes_y,
    # )

    # # Visualize
    # for poly in infill_x:
    #     viewer.plotter.add_mesh(
    #         pv.lines_from_points(poly, close=False),
    #         color="blue", line_width=2
    #     )

    # for poly in infill_y:
    #     viewer.plotter.add_mesh(
    #         pv.lines_from_points(poly, close=False),
    #         color="red", line_width=2
    #     )

    # Part surface = the part's outer surface, already computed above
    pv_mesh = SlicedMold.solid_mesh_to_pyvista(model.mesh)
    linear_mesh = pv_mesh.linear_copy()
    part_surface = linear_mesh.extract_surface(algorithm='dataset_surface')
    part_surface = part_surface.clean(tolerance=1e-6)
    part_surface = part_surface.triangulate()

    sliced_layers = SlicedMold.slice_shadow_volume(
        shadow_volume=shadow_volume,
        part_surface=part_surface,
        layer_height=1.0,
        nozzle_diameter=0.70,
        n_contour_shells=2,
        infill_percentage=5.0,
        plane_z=0.0,
    )

    # Export the printing path of every support layer to a single text
    # file, ready to be loaded by an external trajectory simulator such
    # as RoboDK.
    SlicedMold.export_layers_to_txt(
        sliced_layers,
        DIR / "trajectories" / "mold_path.txt",
        nozzle_diameter=0.70,
        infill_percentage=5.0,
        approach_distance=1.0,
        layer_height=1.0,
        include_orientation=False,
        shell_simplify_tolerance=2,    # mm : shells simplifiés (RDP)
        collapse_infill=True,            # infill : 2 extrémités par ligne
        infill_straight_tolerance=0.3,  # mm
    )

    # colors_infill = {"infill_x": "blue", "infill_y": "cyan"}

    # for layer in sliced_layers[90:91]:
    #     for poly in layer.contour_shells:
    #         viewer.plotter.add_mesh(
    #             pv.lines_from_points(poly, close=False),
    #             color="orange", line_width=3
    #         )
    #     for poly in layer.infill_x:
    #         viewer.plotter.add_mesh(
    #             pv.lines_from_points(poly, close=False),
    #             color="blue", line_width=1
    #         )
    #     for poly in layer.infill_y:
    #         viewer.plotter.add_mesh(
    #             pv.lines_from_points(poly, close=False),
    #             color="cyan", line_width=1
    #         )

    colors = {
        "curve"     : "blue",
        "connection": "red",
        "approach"  : "green",
        "leave"     : "orange"
    }

    print("pre segments")
    for layer in sliced_layers[1:10]:
        segments, types = layer.build_support_path(
            nozzle_diameter=0.70,
            infill_percentage=5.0,
            approach_distance=1.0,
            layer_height=1.0,
        )

        print(f"Layer q={layer.q}: {len(segments)} segments")

        print("pre add mesh")
        for seg, seg_type in zip(segments, types):
            viewer.plotter.add_mesh(
                pv.lines_from_points(seg, close=False),
                color=colors[seg_type],
                line_width=3 if seg_type == "connection" else 1,
            )

    viewer.view(
            show_edges=False,
            show_origin=False,
            f_scaling_factor=10,
            u_scaling_factor=10,
            projection_method=ProjectionMethod.L2,
            opacity=0.3,
            paths=model.fibers_as_list,
        )


if __name__ == "__main__":
    main()
