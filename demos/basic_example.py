import os

# This demo assumes PyNutil is installed (recommended for development):
#   pip install -e .
from PyNutil import PyNutil

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
segmentation_folder = os.path.join(
    repo_root, "tests/test_data/nonlinear_allen_mouse/segmentations/"
)
alignment_json = os.path.join(
    repo_root, "tests/test_data/nonlinear_allen_mouse/alignment.json"
)
colour = [0, 0, 0]
atlas_name = "allen_mouse_25um"
output_folder = os.path.join(repo_root, "test_result/hemi_test_bg6_damage_24_03_2025")

# Initialize PyNutil object
pnt = PyNutil(
    segmentation_folder=segmentation_folder,
    alignment_json=alignment_json,
    colour=colour,
    atlas_name=atlas_name,
    custom_region_path=os.path.join(
        repo_root,
        "tests/test_data/nonlinear_allen_mouse/CustomRegions_fromQCAlign.txt",
    ),
)

# Execute workflow
pnt.get_coordinates(object_cutoff=0, use_flat=False)
pnt.quantify_coordinates()
pnt.interpolate_volume()
pnt.save_analysis(output_folder)
