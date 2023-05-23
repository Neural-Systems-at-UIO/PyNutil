import numpy as np
import pandas as pd
import struct

# related to counting and load
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


# related to counting_and_load
def PixelCountPerRegion(labelsDict, label_colours): 
    """Function for counting no. of pixels per region and writing to CSV based on 
    a dictionary with the region as the key and the points as the value, """
    counted_labels, label_counts = np.unique(labelsDict, return_counts=True)
    # which regions have pixels, and how many pixels are there per region
    counts_per_label = list(zip(counted_labels,label_counts))
    # create a list of unique regions and pixel counts per region

    df_counts_per_label = pd.DataFrame(counts_per_label, columns=["idx","pixel_count"])
    # create a pandas df with regions and pixel counts

    df_label_colours =pd.read_csv(label_colours, sep=",")
    # find colours corresponding to each region ID and add to the pandas dataframe

    #look up name, r, g, b in df_allen_colours in df_counts_per_label based on "idx"
    new_rows = []
    for index, row in df_counts_per_label.iterrows():
        mask = df_label_colours["idx"] == row["idx"] 
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


"""read flat file and write into an np array"""
def flat_to_array(flatfile):
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


# import flat files, count pixels per label, np.unique... etc. nitrc.org/plugins/mwiki/index.php?title=visualign:Deformation

"""
   base=slice["filename"][:-4]
   
   import struct
   with open(base+".flat","rb") as f:
       b,w,h=struct.unpack(">BII",f.read(9))
       data=struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))
"""
