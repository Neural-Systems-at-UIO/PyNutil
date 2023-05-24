from PyNutil import PyNutil

pnt = PyNutil(settings_file=r"test/test4_2017.json")
pnt.build_quantifier()

pnt.get_coordinates()

pnt.quantify_coordinates()

pnt.save_analysis("outputs/test4_2017")
