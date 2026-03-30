"""
Coronal cross-sections with expression heatmap
===============================================

Interpolate pixel values into atlas space with PyNutil, then display
a grid of coronal sections with the Allen STPT structural template as
background and the expression heatmap overlaid in magma.
"""

import numpy as np
import matplotlib.pyplot as plt
from brainglobe_atlasapi import BrainGlobeAtlas

import PyNutil as pnt

IMAGE_DIR = "/home/harryc/github/allen_download_utilities/downloaded_data/05-2788/71717640/expression_25um"
ALIGNMENT_JSON = "/home/harryc/github/PyNutil/tests/test_data/7171717640/05-2788.json"

N_SECTIONS = 1

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
    return_orientation="asr",
)

expr = result.value.astype(np.float32)
expr = np.nan_to_num(expr, nan=0.0)

# Atlas STPT template — "asr" orientation matches the volume directly
stpt = atlas.reference.astype(np.float32)
stpt /= stpt.max()

n_ap = expr.shape[0]
indices = [int(n_ap * 0.5)]

fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

for ax, idx in zip([ax], indices):
    ax.imshow(stpt[idx], cmap="gray", vmin=0, vmax=1)
    ax.imshow(expr[idx], cmap="magma", alpha=0.6,
              vmin=0, vmax=expr.max() * 0.8)
    ax.set_title(f"AP {idx * 25} µm", fontsize=8)
    ax.axis("off")

fig.suptitle("Calbindin-1 expression — coronal sections", fontsize=12)
plt.tight_layout()
plt.savefig(
    "/home/harryc/github/PyNutil/docs/assets/gallery/calb1_cross_section.png",
    bbox_inches="tight",
)
plt.show()
