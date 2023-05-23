"""Create workflow for calculating load based on atlas maps and segmentations"""

import struct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from read_and_write import FlattoArray, LabeltoArray

base= r"../test_data/ttA_2877_NOP_atlasmaps/2877_NOP_tTA_lacZ_Xgal_s037_nl.flat"
label = r"../annotation_volumes\allen2017_colours.csv"

image_arr = FlattoArray(base)

plt.imshow(FlattoArray(base))

"""assign label file values into image array""" 
labelfile = pd.read_csv(r"../annotation_volumes\allen2017_colours.csv")
allen_id_image = np.zeros((h,w)) # create an empty image array
coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
values = image_arr[coordsx, coordsy] # assign x,y coords from image_array into values
lbidx = labelfile['idx'].values
allen_id_image = lbidx[values.astype(int)] # assign allen IDs into image array

plt.imshow(allen_id_image)

"""count pixels for unique idx"""
unique_ids, counts = np.unique(allen_id_image, return_counts=True)

area_per_label = list(zip(unique_ids,counts))
# create a list of unique regions and pixel counts per region

df_area_per_label = pd.DataFrame(area_per_label, columns=["idx","area_count"])
# create a pandas df with regions and pixel counts

print(df_area_per_label)
df_area_per_label.to_csv("../outputs/s037_area_per_idx.csv", sep=";", na_rep='', index= False)




# Count area per unique label
# Scale up to size of corresponding segmentation
# calculate segmentation value per slice
# divide segmentation value per idx per slice by area per idx per slice
# output reports
# also do for whole brain

#Also to do:
# fix position of clear label in label files
