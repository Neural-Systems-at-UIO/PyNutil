from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"test/PVMouse_81264_test.json")
# pnt = PyNutil(settings_file=r"test/test3.json")
# pnt.build_quantifier()

pnt.get_coordinates(object_cutoff=0)

pnt.quantify_coordinates()

pnt.save_analysis("outputs/test4_2017")
