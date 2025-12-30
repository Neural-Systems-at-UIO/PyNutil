import os

# This demo assumes PyNutil is installed (recommended for development):
#   pip install -e .
from PyNutil import PyNutil

###PyNutil is a toolkit for quantifying neuroscientific data using brain atlases
###Here we define a quantifier object
###The segmentations should be images which come out of ilastix, segmenting an object of interest
###The alignment json should be out of DeepSlice, QuickNII, or VisuAlign, it defines the sections position in an atlas
###The colour says which colour is the object you want to quantify in your segmentation. It is defined in RGB
###Finally the atlas name is the relevant atlas from brainglobe_atlasapi you wish to use in Quantification.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))

pnt = PyNutil(settings_file=os.path.join(repo_root, "tests/test_cases/brainglobe_atlas.json"))
pnt.get_coordinates(object_cutoff=0)
pnt.quantify_coordinates()
pnt.save_analysis(os.path.join(repo_root, "demo_data/PyNutil_nonlinear_noflat"))
