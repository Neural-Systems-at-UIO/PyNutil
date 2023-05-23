# pandas is used for working with csv files
import pandas as pd

# nrrd just lets us open nrrd files
import nrrd
import numpy as np
import csv
import json

from datetime import datetime

# import json, use to define input parameters
with open("../test/test5_NOP_s037.json", "r") as f:
    input = json.load(f)
# print(input)

# import our function for converting a folder of segmentations to points
from coordinate_extraction import FolderToAtlasSpace, FolderToAtlasSpaceMultiThreaded
from read_and_write import SaveDataframeasCSV, WritePointsToMeshview, FilesinDirectory
from counting_and_load import PixelCountPerRegion, labelPoints

startTime = datetime.now()

# now we can use our function to convert the folder of segmentations to points
points = FolderToAtlasSpaceMultiThreaded(
    input["segmentation_folder"],
    input["alignment_json"],
    pixelID=input["colour"],
    nonLinear=input["nonlinear"],
)

time_taken = datetime.now() - startTime
print(f"Folder to atlas took: {time_taken}")
# first we need to find the label file for the volume
# then the path to the volume

# read the label files
label_df = pd.read_csv(input["label_path"])
# read the annotation volume, it also has a header but we don't need it
data, header = nrrd.read(input["volume_path"])
# now we can get the labels for each point
labels = labelPoints(points, data, scale_factor=2.5)

# save points to a meshview jsonv
WritePointsToMeshview(points, labels, input["points_json_path"], label_df)

df_counts_per_label_name = PixelCountPerRegion(labels, input["label_path"])
SaveDataframeasCSV(df_counts_per_label_name, input["counts_per_label_name"])

time_taken = datetime.now() - startTime

print(f"overall time taken was: {time_taken}")

# while we havent added it here it would be good to next quantify the number of cells for each label.
# get centroids and areas returns a list of objects and the center coordinate.
# we need to deform the center coordinate according to visualign deformations¨
# we need to then transform the coordinate into atlas space
# and then save labels like before.


# Create a list of flat file names from a directory:

# flatfiles = FilesinDirectory('../test_data/ttA_2877_NOP_atlasmaps')
# print(flatfiles)
