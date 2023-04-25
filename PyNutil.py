
import numpy as np
from DeepSlice.coord_post_processing.spacing_and_indexing import number_sections
import json
from VisuAlignWarpVec import triangulate,  forwardtransform_vec, transform_vec
from glob import glob
from tqdm import tqdm
import cv2
from skimage import measure
import pandas as pd

def getCentroidsAndArea(Segmentation, pixelCutOff=0):
    SegmentationBinary = ~np.all(Segmentation==255, axis=2) 
    labels = measure.label(SegmentationBinary)
    #this finds all the objects in the image
    labelsInfo = measure.regionprops(labels)
    #remove objects that are less than pixelCutOff
    labelsInfo = [label for label in labelsInfo if label.area > pixelCutOff]
    centroids = np.array([label.centroid for label in labelsInfo])
    area = np.array([label.area for label in labelsInfo])
    return centroids, area

def assignPointsToRegions(points, regionVolume):
    Regions = regionVolume[points[:,0].astype(int), points[:,1].astype(int), points[:,2].astype(int)]
    regionDict = {region: [points[Regions==region]] for region in np.unique(Regions)}
    return regionDict


def transformToRegistration(SegHeight, SegWidth, RegHeight, RegWidth):
    Yscale = RegHeight/SegHeight
    Xscale = RegWidth/SegWidth
    return  Yscale,Xscale


def findMatchingPixels(Segmentation, id):
    mask = Segmentation==id
    mask = np.all(mask, axis=2)
    id_positions = np.where(mask)
    idY, idX  = id_positions[0], id_positions[1]
    return idY,idX

def scalePositions(idY, idX, Yscale, Xscale):
    idY = idY * Yscale
    idX = idX * Xscale
    return  idY,idX

def transformToAtlasSpace(anchoring, Y, X, RegHeight, RegWidth):
    O = anchoring[0:3]
    U = anchoring[3:6]
    # swap order of U
    U = np.array([U[0], U[1], U[2]])
    V = anchoring[6:9]
    # swap order of V
    V = np.array([V[0], V[1], V[2]])
    #scale X and Y to between 0 and 1 using the registration width and height
    Yscale = Y/RegHeight
    Xscale = X/RegWidth
    # print("width: ", RegWidth, " height: ", RegHeight, " Xmax: ", np.max(X), " Ymax: ", np.max(Y), " Xscale: ", np.max(Xscale), " Yscale: ", np.max(Yscale))
    XYZV = np.array([Yscale*V[0], Yscale*V[1], Yscale*V[2]])
    XYZU = np.array([Xscale*U[0], Xscale*U[1], Xscale*U[2]])
    O = np.reshape(O, (3,1))
    return (O+XYZU+XYZV).T

def loadVisuAlignJson(filename):
    with open(filename) as f:
        vafile = json.load(f)
    slices = vafile["slices"]
    return slices

def SegmentationToAtlasSpace(slice, SegmentationPath, pixelID='auto', nonLinear=True):
    Segmentation = cv2.imread(SegmentationPath)

    if pixelID == 'auto':
        #remove the background from the segmentation
        SegmentationNoBackGround = Segmentation[~np.all(Segmentation==255, axis=2)]
        pixelID = np.vstack({tuple(r) for r in SegmentationNoBackGround.reshape(-1,3)})#remove background
        #currently only works for a single label
        pixelID = pixelID[0]
    ID_pixels = findMatchingPixels(Segmentation, pixelID)
    #transform pixels to registration space (the registered image and segmentation have different dimensions)
    SegHeight = Segmentation.shape[0]
    SegWidth  = Segmentation.shape[1]
    RegHeight = slice["height"]
    RegWidth  = slice["width"]
    #this calculates reg/seg
    Yscale , Xscale = transformToRegistration(SegHeight,SegWidth,  RegHeight,RegWidth)
    #this creates a triangulation using the reg width
    triangulation   = triangulate(RegWidth, RegHeight, slice["markers"])
    #scale the seg coordinates to reg/seg
    scaledY,scaledX = scalePositions(ID_pixels[0], ID_pixels[1], Yscale, Xscale)
    if nonLinear:
        newX, newY = transform_vec(triangulation, scaledX, scaledY)
    else:
        newX, newY = scaledX, scaledY
    #scale U by Uxyz/RegWidth and V by Vxyz/RegHeight
    points = transformToAtlasSpace(slice['anchoring'], newY, newX, RegHeight, RegWidth)
    # points = points.reshape(-1)
    return points


def FolderToAtlasSpace(folder, QUINT_alignment, pixelID=[0, 0, 0], nonLinear=True):
    slices = loadVisuAlignJson(QUINT_alignment)
    points = []
    segmentationFileTypes = [".png", ".tif", ".tiff", ".jpg", ".jpeg"] 
    Segmentations = [file for file in glob(folder + "*") if any([file.endswith(type) for type in segmentationFileTypes])]
    SectionNumbers = number_sections(Segmentations)
    #order segmentations and sectionNumbers
    # Segmentations = [x for _,x in sorted(zip(SectionNumbers,Segmentations))]
    # SectionNumbers.sort()
    for  SegmentationPath in Segmentations:
        seg_nr = int(number_sections([SegmentationPath])[0])
        current_slice_index = np.where([s["nr"]==seg_nr for s in slices])
        current_slice = slices[current_slice_index[0][0]]
        ##this converts the segmentation to a point cloud
        points.extend(SegmentationToAtlasSpace(current_slice, SegmentationPath, pixelID, nonLinear))
    return points

def WritePoints(pointsDict, filename, infoFile):
    meshview = [
    {
        "idx": idx,
        "count": len(pointsDict[name])//3,
        "name"  : str(name),
        "triplets": str(infoFile["name"].values[0]),
        "r": str(infoFile["r"].values[0]),
        "g": str(infoFile["g"].values[0]),
        "b": str(infoFile["b"].values[0])
    }
    for name, idx in zip(pointsDict.keys(), range(len(pointsDict.keys())))
    ]
    #write meshview json
    with open(filename, "w") as f:
        json.dump(meshview, f)


def labelPoints(points, label_volume, scale_factor=1):
    #first convert the points to 3 columns
    points = np.reshape(points, (-1,3))
    #scale the points
    points = points * scale_factor
    #round the points to the nearest whole number
    points = np.round(points).astype(int)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    #get the label value for each point
    labels = label_volume[x,y,z]
    return labels
