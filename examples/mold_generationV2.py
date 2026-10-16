"""Lightweight visualization of the mold slicing planes.

Instead of building (and plotting) every printing layer like
mold_generation.py does, this example only displays the mold's slicing
planes: the two families of vertical planes (normal to X and normal to
Y) used by the infill, clipped to the shadow volume. Only the part of
each plane lying inside the shadow volume is drawn, as a filled patch.

All the plotting logic lives in slicer.utils.visualization; this script
only prepares the model and calls it.
"""

from pathlib import Path

from oamc.constants import BANNER
from oamc.core import CompositeMaterial, CompositeModel
from oamc.fem.material import IsotropicMaterial, TransverselyIsotropicMaterial
from oamc.integrations.ansys.parser import APDLParser
from oamc.logging import enable_logging

from slicer import SlicedMold, SlicedPiece
from slicer.utils.visualization import plot_slicing_planes

DIR = Path(__file__).parent.resolve()

NOZZLE_DIAMETER = 0.70
INFILL_PERCENTAGE = 5.0
PLANE_Z = 0.0
OFFSET_FACES = 0.05


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

    plot_slicing_planes(
        shadow_volume,
        nozzle_diameter=NOZZLE_DIAMETER,
        infill_percentage=INFILL_PERCENTAGE,
    )


if __name__ == "__main__":
    main()
