import os

from brainglobe_atlasapi import BrainGlobeAtlas
import PyNutil as pnt

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
segmentation_folder = "/mnt/c/users/harryc/Downloads/registration_tools/images/already_registered/segmentations_downsampled/"
alignment_json = "/mnt/c/users/harryc/Downloads/registration_tools/images/already_registered/thumbnails/onlyCalbJSON.json"
colour = [0, 0, 0]
output_folder = "/mnt/c/users/harryc/Downloads/registration_tools/images/already_registered/output"

# Load atlas (BrainGlobe) and alignment
atlas = BrainGlobeAtlas("demba_allen_seg_dev_mouse_p15_20um")
alignment = pnt.read_alignment(alignment_json)
for i in alignment.slices:
    i.width = i.width // 10
    i.height = i.height // 10
# Extract coordinates from segmentations
coords = pnt.seg_to_coords(
    segmentation_folder,
    alignment,
    atlas,
    pixel_id=colour,
    object_cutoff=0,
    segmentation_format="binary",
)

# Quantify by atlas region
label_df = pnt.quantify_coords(coords, atlas)
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
)
