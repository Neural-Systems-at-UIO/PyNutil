import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyNutil import PyNutil

###PyNutil is a toolkit for quantifying neuroscientific data using brain atlases
###Here we define a quantifier object
###The segmentations should be images which come out of ilastix, segmenting an object of interest
###The alignment json should be out of DeepSlice, QuickNII, or VisuAlign, it defines the sections position in an atlas
###The colour says which colour is the object you want to quantify in your segmentation. It is defined in RGB
###The atlas_path is the path to the relevant atlas.nrrd
###The label_path is the path to the corresponding atlas .csv
###The object_cutoff is a cut-off for min object size

pnt = PyNutil(
    segmentation_folder="../tests/test_data/nonlinear_allen_mouse/",
    alignment_json="../tests/test_data/nonlinear_allen_mouse/alignment.json",
    colour=[0, 0, 0],
    atlas_path="../tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    label_path="../tests/test_data/allen_mouse_2017_atlas//allen2017_colours.csv",
)
pnt.get_coordinates(object_cutoff=0)
pnt.quantify_coordinates()
pnt.save_analysis("../demo_data/test_nonlinear_allen_mouse")
