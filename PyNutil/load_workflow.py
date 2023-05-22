import struct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from read_and_write import FlattoArray

base= r"../test_data/ttA_2877_NOP_atlasmaps/2877_NOP_tTA_lacZ_Xgal_s037_nl.flat"

image_arr = FlattoArray(base)

plt.imshow(FlattoArray(base))

"""assign label file values into image array""" 
labelfile = pd.read_csv("../annotation_volumes/allen2017_colours.csv")
allen_id_image = np.zeros((h,w)) # create an empty image array
coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
values = image_arr[coordsx, coordsy] # assign x,y coords from image_array into values
lbidx = labelfile['idx'].values
allen_id_image = lbidx[values.astype(int)] # assign allen IDs into image array

plt.imshow(allen_id_image)