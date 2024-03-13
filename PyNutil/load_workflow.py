"""Create workflow for calculating load based on atlas maps and segmentations"""

import pandas as pd
import cv2

# from read_and_write import flat_to_array, label_to_array
from counting_and_load import flat_to_dataframe

base = r"../test_data/tTA_2877_NOP_s037_atlasmap/2877_NOP_tTA_lacZ_Xgal_s037_nl.flat"
label = r"../annotation_volumes\allen2017_colours.csv"
##optional
seg = r"../test_data/tTA_2877_NOP_s037_seg/2877_NOP_tTA_lacZ_Xgal_resize_Simple_Seg_s037.png"
segim = cv2.imread(seg)
# the indexing [:2] means the first two values and [::-1] means reverse the list
segXY = segim.shape[:2][::-1]
# image_arr = flat_to_array(base, label)

# plt.imshow(flat_to_array(base, label))

df_area_per_label = flat_to_dataframe(base, label, segXY)

"""count pixels in np array for unique idx, return pd df"""
# unique_ids, counts = np.unique(allen_id_image, return_counts=True)

# area_per_label = list(zip(unique_ids, counts))
# create a list of unique regions and pixel counts per region

# df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "area_count"])
# create a pandas df with regions and pixel counts


"""add region name and colours corresponding to each idx into dataframe.
This could be a separate function"""

df_label_colours = pd.read_csv(label, sep=",")
# find colours corresponding to each region ID and add to the pandas dataframe

# look up name, r, g, b in df_allen_colours in df_area_per_label based on "idx"
new_rows = []
for index, row in df_area_per_label.iterrows():
    mask = df_label_colours["idx"] == row["idx"]
    current_region_row = df_label_colours[mask]
    current_region_name = current_region_row["name"].values
    current_region_red = current_region_row["r"].values
    current_region_green = current_region_row["g"].values
    current_region_blue = current_region_row["b"].values

    row["name"] = current_region_name[0]
    row["r"] = current_region_red[0]
    row["g"] = current_region_green[0]
    row["b"] = current_region_blue[0]

    new_rows.append(row)

df_area_per_label_name = pd.DataFrame(new_rows)

print(df_area_per_label_name)
df_area_per_label_name.to_csv(
    "../outputs/NOP_s037_regionareas.csv", sep=";", na_rep="", index=False
)


# Count area per unique label in one flat file - done.
# Scale up to size of corresponding segmentation/ or size of reference atlas if points are already scaled?
# divide "segmentation value per idx per slice" by "area per idx per slice"
# also do for whole brain - need to loop through and match up section with corresponding atlasmap
# output reports per brain and per slice
