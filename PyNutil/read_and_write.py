import json

#related to read and write: loadVisuAlignJson
def loadVisuAlignJson(filename):
    with open(filename) as f:
        vafile = json.load(f)
    slices = vafile["slices"]
    return slices


#related to read and write: WritePoints
def WritePoints(pointsDict, filename, infoFile):
    """write a series of points to a meshview json file. pointsDict is a dictionary with the points.
    pointsDict is created by createRegionDict. infoFile is a csv file with the information about the regions"""
    meshview = [
    {
        "idx": idx,
        "count": len(pointsDict[name])//3,
        "name"  :str(infoFile["name"].values[infoFile["allenID"]==name][0]),
        "triplets": pointsDict[name],
        "r": str(infoFile["r"].values[infoFile["allenID"]==name][0]),
        "g": str(infoFile["g"].values[infoFile["allenID"]==name][0]),
        "b": str(infoFile["b"].values[infoFile["allenID"]==name][0])
    }
    for name, idx in zip(pointsDict.keys(), range(len(pointsDict.keys())))
    ]
    #write meshview json
    with open(filename, "w") as f:
        json.dump(meshview, f)


# related to read and write: WritePointsToMeshview
# this uses createRegionDict in coordinate_extraction.py
def WritePointsToMeshview(points, pointNames, filename, infoFile):
    """this is the function you call more often as it combines the other functions for writing meshview"""
    regionDict = createRegionDict(points, pointNames)
    WritePoints(regionDict, filename, infoFile)