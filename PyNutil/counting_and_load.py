import numpy as np
import pandas as pd
import struct


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
    return df_counts_per_label_name


"""Read flat file and write into an np array"""


def flat_to_array(flat_file):
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
    return image_arr


# Import flat files, count pixels per label, np.unique... etc. nitrc.org/plugins/mwiki/index.php?title=visualign:Deformation

"""
   base=slice["filename"][:-4]
   
   import struct
   with open(base+".flat","rb") as f:
       b,w,h=struct.unpack(">BII",f.read(9))
       data=struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))
"""
