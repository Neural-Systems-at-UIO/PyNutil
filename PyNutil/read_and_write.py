import json
import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt
import os
import nrrd


# related to read and write
# this function reads a VisuAlign JSON and returns the slices
def load_visualign_json(filename):
    with open(filename) as f:
        vafile = json.load(f)
    slices = vafile["slices"]
    return slices


# related to read_and_write, used in write_points_to_meshview
# this function returns a dictionary of region names
def create_region_dict(points, regions):
    """points is a list of points and regions is an id for each point"""
    region_dict = {
        region: points[regions == region].flatten().tolist()
        for region in np.unique(regions)
    }
    return region_dict


# related to read and write: write_points
# this function writes the region dictionary to a meshview json
def write_points(points_dict, filename, info_file):
    meshview = [
        {
            "idx": idx,
            "count": len(points_dict[name]) // 3,
            "name": str(info_file["name"].values[info_file["idx"] == name][0]),
            "triplets": points_dict[name],
            "r": str(info_file["r"].values[info_file["idx"] == name][0]),
            "g": str(info_file["g"].values[info_file["idx"] == name][0]),
            "b": str(info_file["b"].values[info_file["idx"] == name][0]),
        }
        for name, idx in zip(points_dict.keys(), range(len(points_dict.keys())))
    ]
    # write meshview json
    with open(filename, "w") as f:
        json.dump(meshview, f)


# related to read and write: write_points_to_meshview
# this function combines create_region_dict and write_points functions
def write_points_to_meshview(points, point_names, filename, info_file):
    region_dict = create_region_dict(points, point_names)
    write_points(region_dict, filename, info_file)


# I think this might not need to be its own function :)
def save_dataframe_as_csv(df_to_save, output_csv):
    """Function for saving a df as a CSV file"""
    df_to_save.to_csv(output_csv, sep=";", na_rep="", index=False)


def flat_to_array(flatfile):
    """Read flat file and write into an np array, return array"""
    with open(flatfile, "rb") as f:
        # i dont know what b is, w and h are the width and height that we get from the
        # flat file header
        b, w, h = struct.unpack(">BII", f.read(9))
        # data is a one dimensional list of values
        # it has the shape width times height
        data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))

    # convert flat file data into an array, previously data was a tuple
    imagedata = np.array(data)

    # create an empty image array in the right shape, write imagedata into image_array
    image = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            image[y, x] = imagedata[x + y * w]

    image_arr = np.array(image)
    return image_arr


def label_to_array(label_path, image_array):
    """assign label file values into image array, return array"""
    labelfile = pd.read_csv(label_path)
    allen_id_image = np.zeros((h, w))  # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
    values = image_array[
        coordsx, coordsy
    ]  # assign x,y coords from image_array into values
    lbidx = labelfile["idx"].values
    allen_id_image = lbidx[values.astype(int)]  # assign allen IDs into image array
    return allen_id_image


def files_in_directory(directory):
    """return list of flat file names in a directory"""
    list_of_files = []
    for file in os.scandir(directory):
        if file.path.endswith(".flat") and file.is_file:
            # print(filename.path)
            # newfilename, file_ext = os.path.splitext(filename)
            # print(newfilename)
            filename = os.path.basename(file)
            newfilename, file_ext = os.path.splitext(filename)
            list_of_files.append(newfilename)
    return list_of_files


def read_atlas_volume(atlas_volume_path):
    data, header = nrrd.read(atlas_volume_path)
    return data