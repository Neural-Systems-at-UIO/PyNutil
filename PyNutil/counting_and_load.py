import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
import os
import nrrd


# related to counting and load
def label_points(points, label_volume, scale_factor=1):
    """This function takes a list of points and assigns them to a region based on the region_volume.
    These regions will just be the values in the region_volume at the points.
    It returns a dictionary with the region as the key and the points as the value."""
    # First convert the points to 3 columns
    points = np.reshape(points, (-1, 3))
    # Scale the points
    points = points * scale_factor
    # Round the points to the nearest whole number
    points = np.round(points).astype(int)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # make sure the points are within the volume
    x[x < 0] = 0
    y[y < 0] = 0
    z[z < 0] = 0
    mask = (
        (x > label_volume.shape[0] - 1)
        | (y > label_volume.shape[1] - 1)
        | (z > label_volume.shape[2] - 1)
    )
    x[mask] = 0
    y[mask] = 0
    z[mask] = 0

    # Get the label value for each point
    labels = label_volume[x, y, z]

    return labels


# related to counting_and_load
def pixel_count_per_region(
    labels_dict_points, labeled_dict_centroids, df_label_colours
):
    """Function for counting no. of pixels per region and writing to CSV based on
    a dictionary with the region as the key and the points as the value."""
    if labels_dict_points is not None and labeled_dict_centroids is not None:
        counted_labels_points, label_counts_points = np.unique(
            labels_dict_points, return_counts=True
        )
        counted_labels_centroids, label_counts_centroids = np.unique(
            labeled_dict_centroids, return_counts=True
        )
        # Which regions have pixels, and how many pixels are there per region
        counts_per_label = list(
            zip(counted_labels_points, label_counts_points, label_counts_centroids)
        )
        # Create a list of unique regions and pixel counts per region
        df_counts_per_label = pd.DataFrame(
            counts_per_label, columns=["idx", "pixel_count", "object_count"]
        )
    elif labels_dict_points is None and labeled_dict_centroids is not None:
        counted_labels_centroids, label_counts_centroids = np.unique(
            labeled_dict_centroids, return_counts=True
        )
        # Which regions have pixels, and how many pixels are there per region
        counts_per_label = list(zip(counted_labels_centroids, label_counts_centroids))
        # Create a list of unique regions and pixel counts per region
        df_counts_per_label = pd.DataFrame(
            counts_per_label, columns=["idx", "object_count"]
        )
    elif labels_dict_points is not None and labeled_dict_centroids is None:
        counted_labels_points, label_counts_points = np.unique(
            labels_dict_points, return_counts=True
        )
        # Which regions have pixels, and how many pixels are there per region
        counts_per_label = list(zip(counted_labels_points, label_counts_points))
        # Create a list of unique regions and pixel counts per region
        df_counts_per_label = pd.DataFrame(
            counts_per_label, columns=["idx", "pixel_count"]
        )
    # Create a pandas df with regions and pixel counts

    # df_label_colours = pd.read_csv(label_colours, sep=",")
    # Find colours corresponding to each region ID and add to the pandas dataframe

    # Look up name, r, g, b in df_allen_colours in df_counts_per_label based on "idx"
    new_rows = []
    for index, row in df_counts_per_label.iterrows():
        mask = df_label_colours["idx"] == row["idx"]
        current_region_row = df_label_colours[mask]
        current_region_name = current_region_row["name"].values
        current_region_red = current_region_row["r"].values
        current_region_green = current_region_row["g"].values
        current_region_blue = current_region_row["b"].values

        row["name"] = current_region_name[0]
        row["r"] = current_region_red[0]
        row["g"] = current_region_green[0]
        row["b"] = current_region_blue[0]

        new_rows.append(row)

    df_counts_per_label_name = pd.DataFrame(new_rows)
    
    #Task for Sharon:
    #If you can get the areas per region from the flat file here 
    #you can then use those areas to calculate the load per region here
    # and add to dataframe
    #see messing around pyflat.py

    return df_counts_per_label_name


"""Read flat file and write into an np array"""
"""Read flat file, write into an np array, assign label file values, return array"""

def flat_to_array(flat_file, labelfile):
    with open(flat_file, "rb") as f:
        # I don't know what b is, w and h are the width and height that we get from the
        # flat file header
        b, w, h = struct.unpack(">BII", f.read(9))
        # Data is a one dimensional list of values
        # It has the shape width times height
        data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))

    # Convert flat file data into an array, previously data was a tuple
    image_data = np.array(data)

    # Create an empty image array in the right shape, write image_data into image_array
    image = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            image[y, x] = image_data[x + y * w]

    image_arr = np.array(image)
    #return image_arr

    """assign label file values into image array"""
    labelfile = pd.read_csv(labelfile)
    allen_id_image = np.zeros((h, w))  # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
    values = image_arr[coordsx, coordsy]  # assign x,y coords from image_array into values
    lbidx = labelfile["idx"].values
    allen_id_image = lbidx[values.astype(int)]
    return allen_id_image

#def count_per_uniqueidx()

    """count pixels for unique idx"""
    unique_ids, counts = np.unique(allen_id_image, return_counts=True)

    area_per_label = list(zip(unique_ids, counts))
    # create a list of unique regions and pixel counts per region

    df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "area_count"])
    # create a pandas df with regions and pixel counts
    return(df_area_per_label)

# Import flat files, count pixels per label, np.unique... etc. nitrc.org/plugins/mwiki/index.php?title=visualign:Deformation

"""
   base=slice["filename"][:-4]
   
   import struct
   with open(base+".flat","rb") as f:
       b,w,h=struct.unpack(">BII",f.read(9))
       data=struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))
"""

