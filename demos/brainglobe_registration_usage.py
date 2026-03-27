"""
If you have registered your data using BrainGlobe registration
(https://github.com/brainglobe/brainglobe-registration), you can
use these data with PyNutil. At the moment brainglobe registration
outputs one directory per image. Specify the path to the registration
json which is saved by brainglobe registration. PyNutil assumes the
deformation files are saved into the same folder. Everything else
works as normal.
"""
from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(
    "tests/test_data/brainglobe_registration/brainglobe-registration.json"
)

coords = pnt.seg_to_coords(
    pnt.read_segmentation_dir("tests/test_data/brainglobe_segmentation_test_image/", pixel_id=[0, 0, 0]),
    alignment,
    atlas,
)
label_df = pnt.quantify_coords(coords, atlas)

pnt.save_analysis(
    "demo_data/outputs/brainglobe_registration_intensity",
    coords,
    atlas,
    label_df=label_df,
)
