import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyNutil import PyNutil

###PyNutil is a toolkit for quantifying neuroscientific data using brain atlases
###Here we define a quantifier object
###The segmentations should be images which come out of ilastix, segmenting an object of interest
###The alignment json should be out of DeepSlice, QuickNII, or VisuAlign, it defines the sections position in an atlas
###The colour says which colour is the object you want to quantify in your segmentation. It is defined in RGB
###Finally the atlas name is the relevant atlas from brainglobe_atlasapi you wish to use in Quantification.
### get_coordinates, if use_flat=True, place flat files in folder titled "flat_files" at same level as "segmentations" folder .flat or .seg

pnt = PyNutil(
    segmentation_folder="../tests/test_data/nonlinear_allen_mouse/",
    alignment_json="../tests/test_data/nonlinear_allen_mouse/alignment.json",
    colour=[0, 0, 0],
    atlas_name="allen_mouse_25um",
)
pnt.get_coordinates(object_cutoff=0, use_flat=True)
pnt.quantify_coordinates()
