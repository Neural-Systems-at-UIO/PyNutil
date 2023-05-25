from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"test/test5_NOP_s037.json")
# pnt.build_quantifier()

pnt.get_coordinates()

pnt.quantify_coordinates()

pnt.save_analysis("outputs/test5_NOP_25")
