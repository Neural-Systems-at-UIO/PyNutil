import PyNutil

# define parameters
# specify loacation of segmentation folder
segmentation_folder = r"blabla/blabla"
# specify location of json file
json_file = r"blabla/blabla.json"
# specify colour to quantify
colour = [255, 255, 255]
# specify output location
output_path = r"blabla/blabla/output"

quantifier = PyNutil(segmentation_folder, json_file, colour, output_path)

quantifier.build_quantifier()

# define your mask as either a png, or a qcalign damage map
# this mask will be applied to all
quantifier.load_mask(mask_path=r"blablabla/")

# load a custom region file
quantifier.load_custom_regions(custom_region_json=r"blablabla/")
# run coordinate extraction
# ideally extract coordinates per section and whole brain
points = quantifier.get_coordinates()

quantifier.save_coordinates()

objects = quantifier.get_objects()

loads = quantifier.get_loads()

quantifier.save_segmentation_atlas_overlays()
