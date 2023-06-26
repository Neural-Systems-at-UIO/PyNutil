from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"test/test9_NOP_all.json")
# pnt = PyNutil(settings_file=r"test/test7_PyNutil.json")
# pnt.build_quantifier()

pnt.get_coordinates(object_cutoff=0, multi_threaded=False)

pnt.quantify_coordinates()

pnt.save_analysis("outputs/test9_NOP_all")

# remove name, r, g, b, from pixel_
# add to region_areas df