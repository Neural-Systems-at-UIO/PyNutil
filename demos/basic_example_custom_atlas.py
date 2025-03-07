import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyNutil import PyNutil
import os

###PyNutil is a toolkit for quantifying neuroscientific data using brain atlases
###Here we define a quantifier object
###The segmentations should be images which come out of ilastix, segmenting an object of interest
###The alignment json should be out of DeepSlice, QuickNII, or VisuAlign, it defines the sections position in an atlas
###The colour says which colour is the object you want to quantify in your segmentation. It is defined in RGB
###The atlas_path is the path to the relevant atlas.nrrd
###The label_path is the path to the corresponding atlas .csv
###The object_cutoff is a cut-off for min object size
### get_coordinates, if use_flat=True, place flat files in folder titled "flat_files" at same level as "segmentations" folder
# This does not use BrainGlobe API.
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

pnt = PyNutil(
    segmentation_folder=os.path.join(
        script_dir, "../tests/test_data/blank_images/segmentations/"
    ),
    alignment_json=os.path.join(
        script_dir, "../tests/test_data/nonlinear_allen_mouse/alignment.json"
    ),
    colour=[255, 255, 255],
    atlas_path=os.path.join(
        script_dir,
        "../tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    ),
    label_path=os.path.join(
        script_dir, "../tests/test_data/allen_mouse_2017_atlas//allen2017_colours.csv"
    ),

)
pnt.get_coordinates(object_cutoff=0, use_flat=False, non_linear=False)
pnt.quantify_coordinates()
pnt.save_analysis("../test_result/test_nonlinear_allen_mouse_noflat_03_03_25v3")
