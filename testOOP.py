from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"PyNutil/test/test9_PyNutil_linear_only.json")
##Use flat can be set to True if you want to use the flat file
# instead of the visualign json (this is only useful for testing and will be removed)
pnt.get_coordinates(object_cutoff=0, use_flat=False)

pnt.quantify_coordinates()

pnt.save_analysis("PyNutil/outputs/test9_PyNutil_linear_noflat")

# remove name, r, g, b, from pixel_
# add to region_areas df


