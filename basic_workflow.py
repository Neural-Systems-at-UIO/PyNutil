#pandas is used for working with csv files
import pandas as pd
#nrrd just lets us open nrrd files
import nrrd
#import our function for converting a folder of segmentations to points
from PyNutil import FolderToAtlasSpace, labelPoints, WritePointsToMeshview


segmentation_folder = "test_data/oneSection15/"
alignment_json = "test_data/C68_nonlinear.json"
#now we can use our function to convert the folder of segmentations to points
points = FolderToAtlasSpace(segmentation_folder,alignment_json, pixelID=[255, 0, 255], nonLinear=True)
#first we need to find the label file for the volume
label_path = "annotation_volumes//allen2022_colours.csv"
#then the path to the volume
volume_path = "annotation_volumes//annotation_10_reoriented.nrrd"
#read the label file
label_df = pd.read_csv(label_path)
#read the annotation volume, it also has a header but we don't need it
data, header = nrrd.read(volume_path)
#now we can get the labels for each point
labels = labelPoints(points, data, scale_factor=2.5)
#save points to a meshview json
WritePointsToMeshview(points, labels,"outputs/points.json", label_df)
