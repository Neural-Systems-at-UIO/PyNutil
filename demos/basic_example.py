import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyNutil import PyNutil

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_folder = os.path.join(script_dir, "../tests/test_data/nonlinear_allen_mouse/")
alignment_json = os.path.join(script_dir, "../tests/test_data/nonlinear_allen_mouse/alignment.json")
colour = [0, 0, 0]
atlas_name = "allen_mouse_25um"
output_folder = "../test_result/bg_test"

# Initialize PyNutil object
pnt = PyNutil(
    segmentation_folder=segmentation_folder,
    alignment_json=alignment_json,
    colour=colour,
    atlas_name=atlas_name,
)

# Execute workflow
pnt.get_coordinates(object_cutoff=0, use_flat=False)
pnt.quantify_coordinates()
pnt.save_analysis(output_folder)
