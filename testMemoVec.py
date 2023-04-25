import time
startTime = time.time()

import numpy as np
from VisuAlignWarpVec import triangulate,  forwardtransform_vec, forwardtransform
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from ProjectSegmentation3D_vec import FolderToAtlasSpace, getCentroidsAndArea, assignPointsToRegions

import cv2

# Segmentation = cv2.imread("ext-d000033_PVMouseExtraction_pub-Nutil_Quantifier_analysis-81264-Input_dir/ext-d000009_PVMouse_81264_Samp1_resize15__s013_thumbnail_FinalSegmentation.png")
#count the number of isolated segments
# Segmentation[~np.all(Segmentation==255, axis=2)] = 0

# centroids, area = getCentroidsAndArea(Segmentation, 4)

    



#open colour txt file
path = "itksnap_label_description.txt"

# convert to pandas dataframe
import pandas as pd
# use " " as separator
#set column names
df = pd.read_csv(path, sep=" ", header=None,  names=["id", "r", "g", "b", "1a", "1b", "1c", "name"])
df[["name", "allenID"]] = df["name"].str.split(' - ', expand=True)
df.to_csv("allen2022_colours.csv", index=False)




# `read the annotation volume annotation_10.nrrd`
import nrrd

data, header = nrrd.read('annotation_10.nrrd')






points = FolderToAtlasSpace("oneSection15/", "C68_nonlinear.json", pixelID=[255, 0, 255], nonLinear=True)


points = np.reshape(points, (-1, 3))


Points10um = points*2.5

SwapData = np.transpose(data, (2,0,1))
SwapData = SwapData[:, ::-1, ::-1]
Regions = SwapData[Points10um[:,0].astype(int), Points10um[:,1].astype(int), Points10um[:,2].astype(int)]
#efficiently allocate points to the corresponding region in a dictionary with region as key
regionDict = {region : [points[Regions==region]] for region in np.unique(Regions)}

#show frequency of regions
meshview = []
points_all = []
idx = 0
for region in tqdm(regionDict):
    temp_points = np.array(regionDict[region])
    temp_points = temp_points.reshape(-1)
    #write meshview json
    if region == 0:
        infoRow = pd.DataFrame([{"allenID": 0, "name": "background", "r": 255, "g": 255, "b": 255}])
    else:
        infoRow = df.loc[df['allenID'] == str(region)]
    points_all.extend(temp_points.tolist())
    # meshview.append({
    #     "idx": str(1),
    #     "count": str(len(temp_points)//3),
    #     # "name"  : str(infoRow["name"].values[0]),
    #     "name":"1",
    #     "triplets": temp_points.tolist(),
    #     # "r": str(infoRow['r'].values[0]),
    #     # "g": str(infoRow['g'].values[0]),
    #     # "b": str(infoRow['b'].values[0])
    #     "r": 0,
    #     "g": 0,
    #     "b": 255
    # })
    # idx += 1
    #write meshview json


    # executionTime = (time.time() - startTime)

    # print('Execution time in seconds: ' + str(executionTime))
    # points = points.reshape(-1)
    # write meshview json
meshview.append({
    "idx": str(1),
    "count": str(len(temp_points)//3),
    # "name"  : str(infoRow["name"].values[0]),
    "name":"1",
    "triplets": points_all,
    # "r": str(infoRow['r'].values[0]),
    # "g": str(infoRow['g'].values[0]),
    # "b": str(infoRow['b'].values[0])
    "r": 0,
    "g": 0,
    "b": 255
})
with open(f"colour_test_nonlin_notenpc.json", "w") as f:
    json.dump(meshview, f)