#pandas is used for working with csv files
import pandas as pd
#nrrd just lets us open nrrd files
import nrrd
#import our function for converting a folder of segmentations to points
from PyNutil import FolderToAtlasSpace



#now we can use our function to convert the folder of segmentations to points
points = FolderToAtlasSpace("test_data/oneSection15/", "test_data/C68_nonlinear.json", pixelID=[255, 0, 255], nonLinear=True)

#first we need to find the label file for the volume
label_path = "annotation_volumes//allen2022_colours.csv"
#then the path to the volume
volume_path = "annotation_volumes//annotation_10.nrrd"
#read the label file
label_df = pd.read_csv(label_path)
#read the annotation volume, it also has a header but we don't need it
data, header = nrrd.read(volume_path)