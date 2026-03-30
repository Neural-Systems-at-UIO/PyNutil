"""
Expression ISH volume in brainrender (no disk I/O)
===================================================

Loads the 05-2788 alignment, reads the 25 µm downsampled expression images,
interpolates pixel values into atlas space with PyNutil, then renders the
resulting volume directly in brainrender — no NIfTI files written.
"""

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from brainrender import Scene, settings
from brainrender.actors import Volume

import PyNutil as pnt

settings.SHADER_STYLE = "default"

IMAGE_DIR = "/home/harryc/github/allen_download_utilities/downloaded_data/05-2788/71717640/expression_25um"
ALIGNMENT_JSON = "/home/harryc/github/PyNutil/tests/test_data/7171717640/05-2788.json"

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(ALIGNMENT_JSON)

image_series = pnt.read_image_dir(IMAGE_DIR)

result = pnt.interpolate_volume(
    image_series=image_series,
    registration=alignment,
    atlas=atlas,
    value_mode="mean",
    segmentation_mode=False,
    intensity_channel="grayscale",
    do_interpolation=True,
)

arr = result.value
vmax = float(arr.max())

scene = Scene(atlas_name="allen_mouse_25um", title="05-2788 expression volume")

volume = Volume(
    arr,
    voxel_size=25,
    cmap="magma",
    as_surface=False,
)

volume.mesh.alpha([
    (0,             0.0),
    (1,             0.0),
    (vmax * 0.1,    0.1),
    (vmax * 0.75,   0.6),
    (vmax,          0.6),
])
volume.mesh.alpha_unit(300)
volume.mesh.mode(0)

scene.add(volume)
scene.render()
