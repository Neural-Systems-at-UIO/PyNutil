"""
Example of using PyNutil with pre-extracted coordinate data.

Instead of providing segmentation images, you can supply a CSV file
with coordinates in image space. PyNutil will transform them through
the full pipeline (scaling, non-linear deformation, atlas anchoring)
to produce 3D atlas-space coordinates and region quantification.

The CSV must have columns: X, Y, image_width, image_height, section number
"""
from PyNutil import PyNutil

pnt = PyNutil(
    coordinate_file="tests/test_data/coordinates/coordinate_data_section_edges.csv",
    alignment_json="tests/test_data/brainglobe_registration/brainglobe-registration.json",
    atlas_name="allen_mouse_25um",
)

pnt.get_coordinates()
pnt.quantify_coordinates()

pnt.save_analysis(
    "demo_data/outputs/coordinate_example", create_visualisations=False
)
