import os

import PyNutil as pnt

# PyNutil is a toolkit for quantifying neuroscientific data using brain atlases.
# This example uses a custom atlas (not BrainGlobe API).

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))

# Load custom atlas
atlas = pnt.load_custom_atlas(
    atlas_path=os.path.join(
        repo_root,
        "tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    ),
    hemi_path=None,
    label_path=os.path.join(
        repo_root, "tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv"
    ),
)

# Load alignment
alignment = pnt.read_alignment(
    os.path.join(repo_root, "tests/test_data/nonlinear_allen_mouse/alignment.json")
)

# Extract coordinates
coords = pnt.seg_to_coords(
    os.path.join(repo_root, "tests/test_data/nonlinear_allen_mouse/segmentations/"),
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
    object_cutoff=0,
    use_flat=False,
)

# Quantify and save
label_df = pnt.quantify_coords(coords, atlas)
pnt.save_analysis(
    os.path.join(repo_root, "test_result/2custom_atlas_hemi_test_24_03_2025"),
    coords,
    atlas,
    label_df=label_df,
)
