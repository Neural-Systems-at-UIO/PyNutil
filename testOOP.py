from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"test/test8_PyNutil_fixed.json")
# pnt = PyNutil(settings_file=r"test/test7_PyNutil.json")
# pnt.build_quantifier()

pnt.get_coordinates(object_cutoff=0)

pnt.quantify_coordinates()

pnt.save_analysis("outputs/test8_PyNutil")

# remove name, r, g, b, from pixel_
# add to region_areas df