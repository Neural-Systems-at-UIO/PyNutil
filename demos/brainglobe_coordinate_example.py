"""
Example of using PyNutil with BrainGlobe registration and
pre-extracted coordinate data.
"""
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(
    "tests/test_data/brainglobe_coordinates/registration/brainglobe-registration.json"
)

coords = pnt.xy_to_coords(
    "tests/test_data/brainglobe_coordinates/coordinates.csv",
    alignment,
    atlas,
)
label_df = pnt.quantify_coords(coords, atlas)

pnt.save_analysis(
    "demo_data/outputs/brainglobe_coordinate_example",
    coords,
    atlas,
    label_df=label_df,
)
