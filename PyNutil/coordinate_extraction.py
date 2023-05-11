
import numpy as np
import pandas as pd
from DeepSlice.coord_post_processing.spacing_and_indexing import number_sections
import json
from read_and_write import loadVisuAlignJson
from object_counting import labelPoints
from visualign_deformations import triangulate, transform_vec
from glob import glob
from tqdm import tqdm
import cv2
from skimage import measure
import threading

#related to coordinate_extraction
def getCentroidsAndArea(Segmentation, pixelCutOff=0):
    """this function returns the center coordinate of each object in the segmentation.
    You can set a pixelCutOff to remove objects that are smaller than that number of pixels"""
    SegmentationBinary = ~np.all(Segmentation==255, axis=2) 
    labels = measure.label(SegmentationBinary)
    #this finds all the objects in the image
    labelsInfo = measure.regionprops(labels)
    #remove objects that are less than pixelCutOff
    labelsInfo = [label for label in labelsInfo if label.area > pixelCutOff]
    #get the centre points of the objects
    centroids = np.array([label.centroid for label in labelsInfo])
    #get the area of the objects
    area = np.array([label.area for label in labelsInfo])
    #get the coordinates for all the pixels in each object
    coords = np.array([label.coords for label in labelsInfo])
    return centroids, area, coords


# related to coordinate extraction
def transformToRegistration(SegHeight, SegWidth, RegHeight, RegWidth):
    """this function returns the scaling factors to transform the segmentation to the registration space"""
    Yscale = RegHeight/SegHeight
    Xscale = RegWidth/SegWidth
    return  Yscale,Xscale


# related to coordinate extraction
def findMatchingPixels(Segmentation, id):
    """this function returns the Y and X coordinates of all the pixels in the segmentation that match the id provided"""
    mask = Segmentation==id
    mask = np.all(mask, axis=2)
    id_positions = np.where(mask)
    idY, idX  = id_positions[0], id_positions[1]
    return idY,idX


#related to coordinate extraction
def scalePositions(idY, idX, Yscale, Xscale):
    """this function scales the Y and X coordinates to the registration space.
     (the Yscale and Xscale are the output of transformToRegistration)"""
    idY = idY * Yscale
    idX = idX * Xscale
    return  idY,idX


#related to coordinate extraction
def transformToAtlasSpace(anchoring, Y, X, RegHeight, RegWidth):
    """transform to atlas space using the QuickNII anchoring vector"""
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


#related to read and write: loadVisuAlignJson
# this has been moved successfully to read_and_write
"""
def loadVisuAlignJson(filename):
    with open(filename) as f:
        vafile = json.load(f)
    slices = vafile["slices"]
    return slices
"""


# related to coordinate extraction
def SegmentationToAtlasSpace(slice, SegmentationPath, pixelID='auto', nonLinear=True):
    """combines many functions to convert a segmentation to atlas space. It takes care
    of deformations"""
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
    scaledY,scaledX = scalePositions(ID_pixels[0], ID_pixels[1], Yscale, Xscale)
    if nonLinear:
        if "markers" in slice:
            #this creates a triangulation using the reg width
            triangulation   = triangulate(RegWidth, RegHeight, slice["markers"])
            newX, newY = transform_vec(triangulation, scaledX, scaledY)
        else:
            print(f"no markers found for " + slice["filename"])
            newX, newY = scaledX, scaledY
    else:
        newX, newY = scaledX, scaledY


    #scale U by Uxyz/RegWidth and V by Vxyz/RegHeight
    points = transformToAtlasSpace(slice['anchoring'], newY, newX, RegHeight, RegWidth)
    # points = points.reshape(-1)
    return np.array(points)


# related to coordinate extraction
def FolderToAtlasSpace(folder, QUINT_alignment, pixelID=[0, 0, 0], nonLinear=True):
    "apply Segmentation to atlas space to all segmentations in a folder"
    slices = loadVisuAlignJson(QUINT_alignment)
    points = []
    segmentationFileTypes = [".png", ".tif", ".tiff", ".jpg", ".jpeg"] 
    Segmentations = [file for file in glob(folder + "/*") if any([file.endswith(type) for type in segmentationFileTypes])]
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
    return np.array(points)


#related to coordinate extraction
def FolderToAtlasSpaceMultiThreaded(folder, QUINT_alignment, pixelID=[0, 0, 0], nonLinear=True):
    "apply Segmentation to atlas space to all segmentations in a folder"
    slices = loadVisuAlignJson(QUINT_alignment)
    
    segmentationFileTypes = [".png", ".tif", ".tiff", ".jpg", ".jpeg"] 
    Segmentations = [file for file in glob(folder + "/*") if any([file.endswith(type) for type in segmentationFileTypes])]
    SectionNumbers = number_sections(Segmentations)
    #order segmentations and sectionNumbers
    # Segmentations = [x for _,x in sorted(zip(SectionNumbers,Segmentations))]
    # SectionNumbers.sort()
    pointsList = [None] * len(Segmentations)
    threads = []
    for  SegmentationPath, index in zip(Segmentations, range(len(Segmentations))):
        seg_nr = int(number_sections([SegmentationPath])[0])
        current_slice_index = np.where([s["nr"]==seg_nr for s in slices])
        current_slice = slices[current_slice_index[0][0]]
        x = threading.Thread(target=SegmentationToAtlasSpaceMultiThreaded, args=(current_slice, SegmentationPath, pixelID, nonLinear, pointsList, index))
        threads.append(x)
        ##this converts the segmentation to a point cloud
    # start threads
    [t.start() for t in threads]
    # wait for threads to finish
    [t.join() for t in threads]
    # flatten pointsList
    points = [item for sublist in pointsList for item in sublist]
    return np.array(points)


# related to coordinate extraction
def SegmentationToAtlasSpaceMultiThreaded(slice, SegmentationPath, pixelID='auto', nonLinear=True, pointsList=None, index=None):
    """combines many functions to convert a segmentation to atlas space. It takes care
    of deformations"""
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

    #scale the seg coordinates to reg/seg
    scaledY,scaledX = scalePositions(ID_pixels[0], ID_pixels[1], Yscale, Xscale)
    if nonLinear:
        if "markers" in slice:
            #this creates a triangulation using the reg width
            triangulation   = triangulate(RegWidth, RegHeight, slice["markers"])
            newX, newY = transform_vec(triangulation, scaledX, scaledY)
        else:
            print(f"no markers found for " + slice["filename"])
            newX, newY = scaledX, scaledY
    else:
        newX, newY = scaledX, scaledY
    #scale U by Uxyz/RegWidth and V by Vxyz/RegHeight
    points = transformToAtlasSpace(slice['anchoring'], newY, newX, RegHeight, RegWidth)
    # points = points.reshape(-1)
    pointsList[index] = np.array(points)


#related to coordinate extraction or object_counting
def createRegionDict(points, regions):
    """points is a list of points and regions is an id for each point"""
    regionDict = {region:points[regions==region].flatten().tolist() for region in np.unique(regions)}
    return regionDict


#related to read and write: WritePoints

def WritePoints(pointsDict, filename, infoFile):

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

def WritePointsToMeshview(points, pointNames, filename, infoFile):
    regionDict = createRegionDict(points, pointNames)
    WritePoints(regionDict, filename, infoFile)



# related to object_counting: labelPoints
def labelPoints(points, label_volume, scale_factor=1):
    """this function takes a list of points and assigns them to a region based on the regionVolume.
    These regions will just be the values in the regionVolume at the points.
    it returns a dictionary with the region as the key and the points as the value"""
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

# related to object_counting
# consider separating out writing to CSV in future
def PixelCountPerRegion(labelsDict, label_colours): 
    """Function for counting no. of pixels per region and writing to CSV based on 
    a dictionary with the region as the key and the points as the value, """
    counted_labels, label_counts = np.unique(labelsDict, return_counts=True)
    # which regions have pixels, and how many pixels are there per region
    counts_per_label = list(zip(counted_labels,label_counts))
    # create a list of unique regions and pixel counts per region

    df_counts_per_label = pd.DataFrame(counts_per_label, columns=["allenID","pixel count"])
    # create a pandas df with regions and pixel counts

    df_label_colours =pd.read_csv(label_colours, sep=",")
    # find colours corresponding to each region ID and add to the pandas dataframe

    #look up name, r, g, b in df_allen_colours in df_counts_per_label based on "allenID"
    new_rows = []
    for index, row in df_counts_per_label.iterrows():
        mask = df_label_colours["allenID"] == row["allenID"] 
        current_region_row = df_label_colours[mask]
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
    return df_counts_per_label_name
    
   
#def SaveDataframeasCSV(df_to_save):
    #df_to_save.to_csv(output_csv, sep=";", na_rep='', index= False)

