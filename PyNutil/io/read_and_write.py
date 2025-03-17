import json
import numpy as np
import struct
import pandas as pd
import os
import nrrd
import re
from .propagation import propagate
import numpy as np
import struct
import cv2
from .reconstruct_dzi import reconstruct_dzi


def open_custom_region_file(path):
    """
    Opens a custom region file created by QCAlign or manually by the user.

    Parameters
    ----------
    path : str
        the path to the TSV or XLSX file containing the custom region mappings.
        If the file extension is not XLSX we will assume it is TSV. By default
        QCAlign exports TSV files with a TXT extension.

    Returns
    ----------
    custom_region_to_dict : dict


    """
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if len(df.columns) < 2:
        raise ValueError("Expected at least two columns in the file.")
    custom_region_names = df.columns[1:].to_list()
    rgb_values = df.iloc[0, :].values[1:]
    try:
        rgb_values = [list(int(i) for i in rgb.split(";")) for rgb in rgb_values]
    except ValueError:
        print("Error: Non integer value found in rgb list")
    atlas_ids = df.iloc[1:, 1:].T.values
    atlas_ids = [[int(j) for j in i if not j is np.nan] for i in atlas_ids]
    new_ids = []
    new_id = 1
    for i in range(len(atlas_ids)):
        if 0 in atlas_ids[i]:
            new_ids.append(0)
        else:
            new_ids.append(new_id)
            new_id += 1
    if 0 not in new_ids:
        new_ids.append(0)
        custom_region_names.append("unlabeled")
        rgb_values.append([0, 0, 0])
        atlas_ids.append([0])

    custom_region_dict = {
        "custom_ids": new_ids,
        "custom_names": custom_region_names,
        "rgb_values": rgb_values,
        "subregion_ids": atlas_ids,
    }
    df = pd.DataFrame(
        {
            "idx": custom_region_dict["custom_ids"],
            "name": custom_region_dict["custom_names"],
            "r": [c[0] for c in custom_region_dict["rgb_values"]],
            "g": [c[1] for c in custom_region_dict["rgb_values"]],
            "b": [c[2] for c in custom_region_dict["rgb_values"]],
        }
    )
    if df["name"].duplicated().any():
        raise ValueError("Duplicate region names found in custom region file.")
    return custom_region_dict, df


def read_flat_file(file):
    """
    Reads a flat file and returns an image array.

    Args:
        file (str): Path to the flat file.

    Returns:
        ndarray: Image array.
    """
    with open(file, "rb") as f:
        b, w, h = struct.unpack(">BII", f.read(9))
        data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))
    image_data = np.array(data)
    image = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            image[y, x] = image_data[x + y * w]
    return image


def read_seg_file(file):
    """
    Reads a segmentation file and returns an image array.

    Args:
        file (str): Path to the segmentation file.

    Returns:
        ndarray: Image array.
    """
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
        print(f"Target atlas: {atlas}")
        codes = [code() for x in range(code())]
        w = code()
        h = code()
        data = []
        while len(data) < w * h:
            data += [codes[byte() if len(codes) <= 256 else code()]] * (code() + 1)
    image_data = np.array(data)
    image = np.reshape(image_data, (h, w))
    return image


def load_segmentation(segmentation_path: str):
    """
    Loads a segmentation from a file.

    Args:
        segmentation_path (str): Path to the segmentation file.

    Returns:
        ndarray: Segmentation array.
    """
    if segmentation_path.endswith(".dzip"):
        return reconstruct_dzi(segmentation_path)
    else:
        return cv2.imread(segmentation_path)


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
    else:
        slices = vafile["slices"]
    if len(slices) > 1:
        slices = propagate(slices)
    gridspacing = vafile["gridspacing"] if "gridspacing" in vafile else None
    return slices, gridspacing


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


# I think this might not need to be its own function :)
def save_dataframe_as_csv(df_to_save, output_csv):
    """Function for saving a df as a CSV file"""
    df_to_save.to_csv(output_csv, sep=";", na_rep="", index=False)


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
