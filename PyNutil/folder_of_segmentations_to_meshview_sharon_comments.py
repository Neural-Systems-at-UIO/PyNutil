
#pandas is used for working with csv files
import pandas as pd
#nrrd just lets us open nrrd files
import nrrd
import numpy as np
import csv

from datetime import datetime

#import our function for converting a folder of segmentations to points
from PyNutil import FolderToAtlasSpace, labelPoints, WritePointsToMeshview

volume_path = "../annotation_volumes//annotation_10_reoriented.nrrd"
data, header = nrrd.read(volume_path)

startTime = datetime.now()

segmentation_folder = "../test_data/tTA_2877_NOP/"
alignment_json = "../test_data/tTA_2877_NOP_horizontal_final_2017.json"
#now we can use our function to convert the folder of segmentations to points
# colour is BGR (not RGB)
points = FolderToAtlasSpace(segmentation_folder,alignment_json, pixelID=[0, 0, 255], nonLinear=True)
#first we need to find the label file for the volume
label_path = "../annotation_volumes//allen2022_colours.csv"
#then the path to the volume

#read the label files
label_df = pd.read_csv(label_path)
#read the annotation volume, it also has a header but we don't need it
#now we can get the labels for each point
labels = labelPoints(points, data, scale_factor=2.5)
#save points to a meshview json
WritePointsToMeshview(points, labels,"../outputs/points.json", label_df)

#Task:
# Make a pandas dataframe
# Column 1: counted_labels
# Column 2: label_counts
# Column 3: region_name (look up by reading Allen2022_colours.csv, look up name and RGB)
# Save dataframe in output as CSV
# next task is to create functions from this. 
counted_labels, label_counts = np.unique(labels, return_counts=True)
counts_per_label = list(zip(counted_labels,label_counts))

df_counts_per_label = pd.DataFrame(counts_per_label, columns=["allenID","pixel count"])

allen_colours = "../annotation_volumes//allen2022_colours.csv"

df_allen_colours =pd.read_csv(allen_colours, sep=",")
df_allen_colours

#look up name, r, g, b in df_allen_colours in df_counts_per_label based on "allenID"
new_rows = []
for index, row in df_counts_per_label.iterrows():
    mask = df_allen_colours["allenID"] == row["allenID"] 
    current_region_row = df_allen_colours[mask]
    current_region_name = current_region_row["name"].values
    current_region_red = current_region_row["r"].values
    current_region_green = current_region_row["g"].values
    current_region_blue = current_region_row["b"].values

    row["name"]  = current_region_name[0]
    row["r"] = current_region_red[0]
    row["g"] = current_region_green[0]
    row["b"] = current_region_blue[0]
    
    new_rows.append(row)

df_counts_per_label_name = pd.DataFrame(new_rows)
#df_counts_per_label_name

# write to csv file
df_counts_per_label_name.to_csv("../outputs/counts_per_allenID.csv", sep=";", na_rep='', index= False)

#r = df_allen_colours["r"]
#g = df_allen_colours["g"]
#b = df_allen_colours["b"]
#region_name = df_allen_colours["name"]

#while we havent added it here it would be good to next quantify the number of cells for each label.

time_taken = datetime.now() - startTime

print(f"time taken was: {time_taken}")


#get centroids and areas returns a list of objects and the center coordinate.

#we need to deform the center coordinate according to visualign deformations¨

#we need to then transform the coordinate into atlas space

#and then save labels like before.