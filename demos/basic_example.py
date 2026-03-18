import os

from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

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
output_folder = os.path.join(repo_root, "test_result/hemi_test_bg6_damage_24_03_2025")

# Load atlas (BrainGlobe) and alignment
atlas = BrainGlobeAtlas("allen_mouse_25um")
alignment = pnt.read_alignment(alignment_json)

# Extract coordinates from segmentations
coords = pnt.seg_to_coords(
    segmentation_folder,
    alignment,
    atlas,
    pixel_id=colour,
    object_cutoff=0,
    use_flat=False,
    segmentation_format="binary",
)

# Quantify by atlas region
label_df, per_section_df = pnt.quantify_coords(coords, atlas)
# Optionally generate a 3D heatmap
pnt.interpolate_volume(
    segmentation_folder=segmentation_folder,
    alignment_json=alignment_json,
    colour=colour,
    atlas=atlas,
)
# Save results
pnt.save_analysis(
    output_folder,
    coords,
    atlas,
    label_df=label_df,
    per_section_df=per_section_df,
)
