"""
Example of using PyNutil with pre-extracted coordinate data.

Instead of providing segmentation images, you can supply a DataFrame
with coordinates in image space. PyNutil will transform them through
the full pipeline (scaling, non-linear deformation, atlas anchoring)
to produce 3D atlas-space coordinates and region quantification.

The DataFrame must have columns: X, Y, image_width, image_height, section number
"""
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(
    "tests/test_data/brainglobe_registration/brainglobe-registration.json"
)

coords = pnt.xy_to_coords(
    pd.read_csv("tests/test_data/coordinates/coordinate_data_section_edges.csv"),
    alignment,
    atlas,
)

label_df = pnt.quantify_coords(coords, atlas)

pnt.save_analysis(
    "demo_data/outputs/coordinate_example",
    coords,
    atlas,
    label_df=label_df,
)
