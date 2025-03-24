import json
import numpy as np
import struct
import pandas as pd
import os
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
def load_visualign_json(filename, apply_damage_mask):
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
    for slice in slices:
        if not apply_damage_mask:
            if "grid" in slice:
                slice.pop("grid")
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
def write_points_to_meshview(points, point_names, hemi_label, filename, info_file):
    if not (hemi_label == None).all():
        split_fn_left = filename.split('/')
        split_fn_left[-1] = "left_hemisphere_" + split_fn_left[-1]
        outname_left = os.sep.join(split_fn_left)
        left_region_dict = create_region_dict(points[hemi_label==1], point_names[hemi_label==1])
        write_points(left_region_dict, outname_left, info_file)
        split_fn_right = filename.split('/')
        split_fn_right[-1] = "right_hemisphere_" + split_fn_right[-1]
        outname_right = os.sep.join(split_fn_right)
        right_region_dict = create_region_dict(points[hemi_label==2], point_names[hemi_label==2])
        write_points(right_region_dict, outname_right, info_file)
    region_dict = create_region_dict(points, point_names)
    write_points(region_dict, filename, info_file)


