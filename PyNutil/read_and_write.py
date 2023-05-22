import json
import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt


#related to read and write
# this function reads a VisuAlign JSON and returns the slices
def loadVisuAlignJson(filename):
    with open(filename) as f:
        vafile = json.load(f)
    slices = vafile["slices"]
    return slices


#related to read_and_write, used in WritePointsToMeshview
# this function returns a dictionary of region names
def createRegionDict(points, regions):
    """points is a list of points and regions is an id for each point"""
    regionDict = {region:points[regions==region].flatten().tolist() for region in np.unique(regions)}
    return regionDict


#related to read and write: WritePoints
# this function writes the region dictionary to a meshview json
def WritePoints(pointsDict, filename, infoFile):

    meshview = [
    {
        "idx": idx,
        "count": len(pointsDict[name])//3,
        "name"  :str(infoFile["name"].values[infoFile["idx"]==name][0]),
        "triplets": pointsDict[name],
        "r": str(infoFile["r"].values[infoFile["idx"]==name][0]),
        "g": str(infoFile["g"].values[infoFile["idx"]==name][0]),
        "b": str(infoFile["b"].values[infoFile["idx"]==name][0])
    }
    for name, idx in zip(pointsDict.keys(), range(len(pointsDict.keys())))
    ]
    #write meshview json
    with open(filename, "w") as f:
        json.dump(meshview, f)


# related to read and write: WritePointsToMeshview
# this function combines createRegionDict and WritePoints functions
def WritePointsToMeshview(points, pointNames, filename, infoFile):
    regionDict = createRegionDict(points, pointNames)
    WritePoints(regionDict, filename, infoFile)


def SaveDataframeasCSV(df_to_save, output_csv):
    """Function for saving a df as a CSV file"""
    df_to_save.to_csv(output_csv, sep=";", na_rep='', index= False)


def FlattoArray(flatfile):
    """Read flat file and write into an np array, return array"""
    with open(flatfile,"rb") as f:
        #i dont know what b is, w and h are the width and height that we get from the 
        #flat file header
        b,w,h=struct.unpack(">BII",f.read(9))
        #data is a one dimensional list of values
        #it has the shape width times height
        data =struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))

    #convert flat file data into an array, previously data was a tuple
    imagedata = np.array(data)

    #create an empty image array in the right shape, write imagedata into image_array
    image = np.zeros((h,w))
    for x in range(w):
        for y in range(h):
            image[y,x] = imagedata[x+y*w]

    image_arr = np.array(image)
    return image_arr


def LabeltoArray(label_path, image_array):
    """assign label file values into image array, return array""" 
    labelfile = pd.read_csv(label_path)
    allen_id_image = np.zeros((h,w)) # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
    values = image_array[coordsx, coordsy] # assign x,y coords from image_array into values
    lbidx = labelfile['idx'].values
    allen_id_image = lbidx[values.astype(int)] # assign allen IDs into image array
    return allen_id_image

