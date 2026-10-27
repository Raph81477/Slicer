# Slicer non planaire (impression 3D composites, bras robotique 6 ddl)

Slicing non planaire de modèles composites (fibres continues) pour
l'impression avec un bras robotique à 6 degrés de liberté. La génération
des trajectoires se fait en Python (PyVista / NumPy / SciPy) ; la
simulation du robot se fait dans RoboDK.

## Structure

- `src/slicer/core/SlicedObject.py` — classe parente (surface d'une
  couche, distance au bord, offset, contours, construction du path,
  export des trajectoires).
- `src/slicer/core/SlicedPiece.py` — classe héritée pour la pièce à
  imprimer (fibres, plans de slicing alignés aux fibres, path
  fiber-aligned + contours).
- `src/slicer/core/SlicedMold.py` — classe héritée pour le moule
  inférieur généré automatiquement (shadow volume, slicing, infill
  zigzag, export).
- `src/slicer/utils/visualization.py` — visualisation du moule par
  plans de slicing (patches remplis, ear-clipping).
- `src/slicer/gui/` — interface graphique Tkinter.

## GUI (Tkinter)

Lancer l'interface :

```bash
python -m slicer.gui
```

Fonctionnalités :

- Choix du fichier .dat (modèle ANSYS) et du dossier des fibres
  (CSV exportés par OAMC).
- Saisie manuelle de `layer_height` et `layer_width` (diamètre buse),
  plus `fiber_diameter` et `n_layers_under_piece`.
- Options « Rotate and lift model » et « Mold generation ». Activer le
  mold active automatiquement le rotate and lift. Le lift place le
  modèle tel que `zmin > 0`, puis l'augmente de
  `n_layers_under_piece * layer_height`. Quand le mold est activé, on
  peut choisir le dossier où les fibres transformées (rotation + lift)
  sont enregistrées.
- Paramètres tunables du path de la pièce : `contour_width`,
  `approach_distance`, `max_connection_length`, `merge_tolerance`,
  `contour_isovalues`.
- Paramètres tunables de l'extraction des edges (utilisée pour le
  calcul des contours) : `feature_angle`, `boundary_edges`,
  `non_manifold_edges`.
- Paramètres du moule : méthode de shadow volume (vertical / cone /
  simple), `infill_percentage`, `n_contour_shells`.
- Visualisation dans une figure PyVista externe, avec options
  séparées : modèle, edges détectés, path d'impression (courbes bleu,
  connections rouge, approach vert, leave orange), mold (visualisation
  par plans de slicing), légende, et sélection des couches à
  afficher (ex. `0-4` ou `2,5,7`).
- Matériaux (fibre transversement isotrope + matrice isotrope)
  modifiables.
- Export des trajectoires (pièce + moule) vers des fichiers texte
  prêts pour RoboDK.

- Onglet « Fiber tuning » : pipeline OAMC complet de génération/optimisation
  des fibres (charge de palier, mold optionnel, optimisation aux moindres
  carrés avec itérations, load paths LPP, post-traitement RDP/outliers,
  export des fibres et CSV de convergence), réutilisé du tuner OAMC
  autonome, dans une page dédiée du slicer.
