"""
Visualising an interpolated volume in brainrender
==================================================

Render a PyNutil-generated NIfTI volume (e.g. from
:func:`PyNutil.save_volume_niftis`) as a semi-transparent volumetric heatmap
in a brainrender scene.

``as_surface=False`` uses VTK volume rendering (a true 3-D fog, not a mesh).
Transparency is controlled by two parameters on the underlying vedo Volume:

* ``alpha([(value, opacity), ...])`` — opacity transfer function.  Zero / low
  values are mapped to 0 (invisible); high values ramp up to the desired peak
  opacity.
* ``alpha_unit(u)`` — opacity unit distance in µm.  This is the distance a
  ray must travel through the volume to accumulate one full unit of opacity.
  For 25 µm voxels the brain is thousands of µm wide, so this must be large
  (hundreds of µm) or every ray will saturate immediately and the volume will
  look solid.  Increase to make the whole volume more transparent.
"""

import nibabel as nib
import numpy as np
from brainrender import Scene, settings
from brainrender.actors import Volume

settings.SHADER_STYLE = "default"  # disable cartoon outlines

img = nib.load("demo_data/Rorb_interp_25um(1).nii.gz")
arr = np.asanyarray(img.dataobj).astype(np.float32)


arr = np.flip(arr).transpose(0, 2, 1).copy()

vmax = float(arr.max())

scene = Scene(atlas_name="ccfv3augmented_mouse_10um", title="Rorb interpolated volume")

volume = Volume(
    arr,
    voxel_size=25,      # µm per voxel
    cmap="magma",
    as_surface=False,   # true VTK volume rendering
)

# ── Opacity transfer function ─────────────────────────────────────────────────
# (scalar_value, opacity) pairs — values between points are interpolated.
# Zero background → invisible; ramp up through signal; cap at 0.6 at peak.
volume.mesh.alpha([
    (0,    0.0),
    (1,    0.0),   # suppress noise just above zero
    (vmax * 0.1, 0.1),
    (vmax * 0.75, 0.6),
    (vmax, 0.6),
])

# ── Opacity unit distance ─────────────────────────────────────────────────────
# A ray travelling this many µm accumulates one "unit" of opacity from the
# transfer function above.  Too small → looks solid.  Increase to see through.
volume.mesh.alpha_unit(300)

# Composite (front-to-back) blending — best for heatmap-style volumes.
volume.mesh.mode(0)

scene.add(volume)
scene.render()
