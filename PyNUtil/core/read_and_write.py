import json
import numpy as np
import struct
import pandas as pd
import os
import nrrd
import re
from .propagation import propagate


# related to read and write
# this function reads a VisuAlign JSON and returns the slices
def load_visualign_json(filename):
    with open(filename) as f:
        vafile = json.load(f)
    if filename.endswith(".waln") or filename.endswith("wwrp"):
        slices = vafile["sections"]
        vafile["slices"] = slices
        for slice in slices:
            slice["nr"] = int(re.search(r"_s(\d+)", slice["filename"]).group(1))
            if "ouv" in slice:
                slice["anchoring"] = slice["ouv"]

        name = os.path.basename(filename)
        lz_compat_file = {
            "name": name,
            "target": vafile["atlas"],
            "target-resolution": [456, 528, 320],
            "slices": slices,
        }

    else:
        slices = vafile["slices"]
    if len(slices) > 1:
        slices = propagate(slices)
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
            "r": int(info_file["r"].values[info_file["idx"] == name][0]),
            "g": int(info_file["g"].values[info_file["idx"] == name][0]),
            "b": int(info_file["b"].values[info_file["idx"] == name][0]),
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


def flat_to_array(file, labelfile):
    """Read flat file, write into an np array, assign label file values, return array"""
    if file.endswith(".flat"):
        with open(file, "rb") as f:
            # I don't know what b is, w and h are the width and height that we get from the
            # flat file header
            b, w, h = struct.unpack(">BII", f.read(9))
            # Data is a one dimensional list of values
            # It has the shape width times height
            data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))
    elif file.endswith(".seg"):
        with open(file, "rb") as f:

            def byte():
                return f.read(1)[0]

            def code():
                c = byte()
                if c < 0:
                    raise "!"
                return c if c < 128 else (c & 127) | (code() << 7)

            if "SegRLEv1" != f.read(8).decode():
                raise "Header mismatch"
            atlas = f.read(code()).decode()
            codes = [code() for x in range(code())]
            w = code()
            h = code()
            data = []
            while len(data) < w * h:
                data += [codes[byte() if len(codes) <= 256 else code()]] * (code() + 1)

    # convert flat file data into an array, previously data was a tuple
    imagedata = np.array(data)

    # create an empty image array in the right shape, write imagedata into image_array
    image = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            image[y, x] = imagedata[x + y * w]

    image_arr = np.array(image)
    # return image_arr

    """assign label file values into image array"""
    labelfile = pd.read_csv(labelfile)
    allen_id_image = np.zeros((h, w))  # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
    values = image_arr[
        coordsx, coordsy
    ]  # assign x,y coords from image_array into values
    lbidx = labelfile["idx"].values
    allen_id_image = lbidx[values.astype(int)]
    return allen_id_image


def label_to_array(label_path, image_array):
    """
    Assign label file values into image array and return the resulting array.

    Args:
        label_path (str): Path to the label file (CSV format).
        image_array (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Array with Allen IDs assigned.

    Raises:
        FileNotFoundError: If the label file is not found.
        ValueError: If the image array is empty or has invalid dimensions.
    """
    try:
        # Check if image_array is valid and notify
        if image_array.size == 0:
            raise ValueError("Input image array is empty.")

        h, w = image_array.shape
        if h == 0 or w == 0:
            raise ValueError("Invalid image dimensions.")

        # Read label file
        try:
            labelfile = pd.read_csv(label_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file not found: {label_path}")
        allen_id_image = np.zeros((h, w), dtype=np.int64)
        coordsy, coordsx = np.meshgrid(np.arange(w), np.arange(h))
        values = image_array[
            coordsx, coordsy
        ]  # Assign x,y coords from image_array into values
        lbidx = labelfile["idx"].values
        allen_id_image = lbidx[
            values.astype(int)
        ]  # Assign Allen IDs to the image array

        return allen_id_image

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while processing the label array: {str(e)}"
        )


def files_in_directory(directory):
    """return list of flat file names in a directory"""
    list_of_files = []
    for file in os.scandir(directory):
        if file.path.endswith(".flat") and file.is_file:
            filename = os.path.basename(file)
            newfilename, file_ext = os.path.splitext(filename)
            list_of_files.append(newfilename)
    return list_of_files


def read_atlas_volume(atlas_volume_path):
    """return data from atlas volume"""
    data, header = nrrd.read(atlas_volume_path)
    return data
