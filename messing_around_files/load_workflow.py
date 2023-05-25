"""Create workflow for calculating load based on atlas maps and segmentations"""

import struct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from read_and_write import FlattoArray, LabeltoArray

base = r"../test_data/tTA_2877_NOP_s037_atlasmap/2877_NOP_tTA_lacZ_Xgal_s037_nl.flat"
label = r"../annotation_volumes\allen2017_colours.csv"

image_arr = FlattoArray(base)

plt.imshow(FlattoArray(base))

"""assign label file values into image array"""
labelfile = pd.read_csv(r"../annotation_volumes\allen2017_colours.csv")
allen_id_image = np.zeros((h, w))  # create an empty image array
coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
values = image_arr[coordsx, coordsy]  # assign x,y coords from image_array into values
lbidx = labelfile["idx"].values
allen_id_image = lbidx[values.astype(int)]  # assign allen IDs into image array

plt.imshow(allen_id_image)

"""count pixels for unique idx"""
unique_ids, counts = np.unique(allen_id_image, return_counts=True)

area_per_label = list(zip(unique_ids, counts))
# create a list of unique regions and pixel counts per region

df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "area_count"])
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
    "../outputs/test5_s037_area_per_idx.csv", sep=";", na_rep="", index=False
)


# Count area per unique label in one flat file - done.
# Scale up to size of corresponding segmentation/ or size of reference atlas if points are already scaled?
# divide "segmentation value per idx per slice" by "area per idx per slice"
# also do for whole brain - need to loop through and match up section with corresponding atlasmap
# output reports per brain and per slice
