import json
import numpy as np


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