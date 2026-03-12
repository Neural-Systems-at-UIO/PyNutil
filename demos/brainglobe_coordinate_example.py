"""
Example of using PyNutil with BrainGlobe registration and
pre-extracted coordinate data.
"""
from PyNutil import PyNutil

pnt = PyNutil(atlas_name="allen_mouse_25um")

pnt.get_coordinates(
    coordinate_file="tests/test_data/brainglobe_coordinates/coordinates.csv",
    alignment_json="tests/test_data/brainglobe_coordinates/brainglobe-registration.json",
)
pnt.quantify_coordinates()

pnt.save_analysis(
    "demo_data/outputs/brainglobe_coordinate_example",
    create_visualisations=False,
)
