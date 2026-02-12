"""
If you have registered your data using BrainGlobe registration
(https://github.com/brainglobe/brainglobe-registration).You can
use these data with PyNutil. At the moment brainglobe registration
outputs one directory per image. Specify the path to the registration
json which is saved by brainglobe registration. PyNutil assumes the
deformation files are saved into the same folder. Everything else
works as normal.
"""
from PyNutil import PyNutil

pnt = PyNutil(
    segmentation_folder="tests/test_data/brainglobe_segmentation_test_image/",
    alignment_json="tests/test_data/brainglobe_registration/brainglobe-registration.json",
    atlas_name="allen_mouse_25um",
    colour=[0,0,0]
)

pnt.get_coordinates()
pnt.quantify_coordinates()

pnt.save_analysis("demo_data/outputs/brainglobe_registration_intensity", create_visualisations=False)
