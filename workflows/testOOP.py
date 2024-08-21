import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyNutil import PyNutil
import os

pnt = PyNutil(settings_file=r"test/test8_PyNutil_bigcaudoputamen.json")

##use_flat can be set to True if you want to use the flat file
## for method select between "all", "per_pixel" and "per_object"
pnt.get_coordinates(object_cutoff=0, method="all", use_flat=False)

pnt.quantify_coordinates()

pnt.save_analysis("../PyNutil/outputs/test9_PyNutil_bigcaudoputamen_new")

# remove name, r, g, b, from pixel_
# add to region_areas df


