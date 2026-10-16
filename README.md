# Slicer

Slicing d'une pièce composite (plastique + fibres de carbone) pour impression 3D non planaire avec un bras robotique 6 ddl : génération des trajectoires en Python (PyVista), simulation du robot dans RoboDK.

## Structure

- `src/slicer/SlicedObject.py` — classe parente : découpage, chemins d'impression, export trajectoire.
- `src/slicer/SlicedPiece.py` — pièce à imprimer : infill aligné fibres + contours.
- `src/slicer/SlicedMold.py` — moule inférieur généré automatiquement.
- `examples/` — scripts d'exemple (génération des couches, moule, programme RoboDK).

## Exemples

### `examples/print_layers.py`

Génère les couches et exporte la trajectoire plastique en CSV, un segment par
fichier : `examples/trajectories/layer_<i>/segment_<j>.csv` avec les colonnes
`x,y,z,type,layer` (`type` ∈ curve / connection / approach / leave).

### `examples/roboDK_program.py`

Génère le programme RoboDK `PrintPlasticFiber` : 1re couche de plastique,
puis 1re couche de fibres par-dessus. Ouvrir `RobotPrint_Printhead.rdk` dans
RoboDK, puis lancer le script (RoboDK doit tourner). Le code KRL
(`anisoprint_*`) n'apparaît que dans le `.src` exporté, pas en simulation.

Attend, à côté du script :
- `trajectories/layer_<i>/segment_<j>.csv` — plastique (export CSV par segment)
- `fibers/layer_<i>/fiber_<i>_<j>.csv` — fibres (`x,y,z,nx,ny,nz`)

Options utiles : `--trajectories`, `--fibers`, `--plastic-layer`, `--fiber-layer`.

### `examples/mold_generation.py`

Génère le moule inférieur automatiquement.
