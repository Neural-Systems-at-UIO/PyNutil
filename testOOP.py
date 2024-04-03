from PyNutil import PyNutil
import os

os.chdir("..")
pnt = PyNutil(settings_file=r"PyNutil/test/test10_PyNutil_web.json")

##Use flat can be set to True if you want to use the flat file
# instead of the visualign json (this is only useful for testing and will be removed)
pnt.get_coordinates(object_cutoff=0, use_flat=False)

pnt.quantify_coordinates()

pnt.save_analysis("PyNutil/outputs/test10_PyNutil_web")

# remove name, r, g, b, from pixel_
# add to region_areas df


