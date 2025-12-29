"""Sometimes you may want to measure the intensity of input images.
To do this we specify image_folder instead of segmentation folder
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyNutil import PyNutil

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(
    script_dir, "../tests/test_data/image_intensity/images/"
)
alignment_json = os.path.join(
    script_dir, "../tests/test_data/image_intensity/alignment.json"
)
atlas_name = "allen_mouse_25um"
output_folder = "../test_result/intensity_measurement"

# Initialize PyNutil object
pnt = PyNutil(
    image_folder=image_folder,
    alignment_json=alignment_json,
    atlas_name=atlas_name,
    custom_region_path=os.path.join(
        script_dir,
        "../tests/test_data/nonlinear_allen_mouse/CustomRegions_fromQCAlign.txt",
    ),
)

# Execute workflow
pnt.get_coordinates()
pnt.quantify_coordinates()
# pnt.interpolate_volume()
pnt.save_analysis(output_folder)
