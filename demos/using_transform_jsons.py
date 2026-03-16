import os

import PyNutil as pnt

# PyNutil is a toolkit for quantifying neuroscientific data using brain atlases.
# The alignment json should be from DeepSlice, QuickNII, or VisuAlign.
# The colour says which colour is the object you want to quantify in your segmentation (RGB).
# The atlas name is the relevant atlas from brainglobe_atlasapi.

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))

atlas = pnt.load_atlas_data("allen_mouse_25um")
alignment = pnt.read_alignment(
    os.path.join(repo_root, "tests/test_data/nonlinear_allen_mouse/alignment.json")
)

coords = pnt.seg_to_coords(
    os.path.join(repo_root, "tests/test_data/nonlinear_allen_mouse/segmentations/"),
    alignment,
    atlas,
    pixel_id=[0, 0, 0],
    object_cutoff=0,
)
label_df, per_section_df = pnt.quantify_coords(coords, atlas)

pnt.save_analysis(
    os.path.join(repo_root, "demo_data/PyNutil_nonlinear_noflat"),
    coords,
    atlas,
    label_df=label_df,
    per_section_df=per_section_df,
)
