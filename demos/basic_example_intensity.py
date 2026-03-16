"""Sometimes you may want to measure the intensity of input images.
To do this we use image_to_coords instead of seg_to_coords.
"""
import os

from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
image_folder = os.path.join(
    repo_root, "tests/test_data/image_intensity/images/"
)
alignment_json = os.path.join(
    repo_root, "tests/test_data/image_intensity/alignment.json"
)
output_folder = os.path.join(repo_root, "test_result/intensity_measurement")

# Load atlas and alignment
atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(alignment_json)

# Extract intensity data
coords = pnt.image_to_coords(
    image_folder,
    alignment,
    atlas,
)

# Quantify by atlas region
label_df, per_section_df = pnt.quantify_coords(coords, atlas)

# Save results
pnt.save_analysis(
    output_folder,
    coords,
    atlas,
    label_df=label_df,
    per_section_df=per_section_df,
)
