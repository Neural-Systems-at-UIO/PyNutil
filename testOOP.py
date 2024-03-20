from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"PyNutil/test/test8_PyNutil_fixed.json")
##Use flat can be set to True if you want to use the flat file
# instead of the visualign json (this is only useful for testing and will be removed)
pnt.get_coordinates(object_cutoff=0, use_flat=True)

pnt.quantify_coordinates()

pnt.save_analysis("PyNutil/outputs/test8_PyNutil_bigcaudoputamen_newatlasmaps")

# remove name, r, g, b, from pixel_
# add to region_areas df


