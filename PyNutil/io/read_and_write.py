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
from .atlas_loader import load_atlas_labels

def open_custom_region_file(path):
    """
    Opens a custom region file (TSV or XLSX) and returns both a dictionary
    of region information and a corresponding pandas DataFrame.

    The dictionary contains:
      - 'custom_ids': The unique IDs for each region.
      - 'custom_names': The region names.
      - 'rgb_values': The RGB values for each region.
      - 'subregion_ids': Lists of underlying atlas IDs for each region.

    The returned DataFrame has columns:
      - idx: The unique IDs for each region.
      - name: The region names.
      - r, g, b: The RGB values for each region.

    Parameters
    ----------
    path : str
        Path to the TSV or XLSX file containing region information.

    Returns
    -------
    dict
        A dictionary with region data fields ('custom_ids', 'custom_names', 'rgb_values', and 'subregion_ids').
    pandas.DataFrame
        A DataFrame summarizing the region IDs, names, and RGB information.
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
    Reads a custom 'flat' file format and returns its contents as a 2D NumPy array.

    This format includes a header encoding bit-depth (B), width (W), and height (H).
    Pixel data follows in a sequence that is unpacked into a 2D array.

    Parameters
    ----------
    file : str
        Path to the flat file to read.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array containing image data.
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
    Reads a segmentation file encoded with a specialized format (SegRLEv1),
    decodes it, and returns a 2D NumPy array representing segment labels.

    The file contains a header with atlas information and compression codes
    that are used to rebuild the segmentation data.

    Parameters
    ----------
    file : str
        Path to the segmentation file.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array with segmentation labels.
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
    Loads segmentation data either from a '.dzip' file or a standard image file.
    If the file ends with '.dzip', it will be processed as a DZI using 'reconstruct_dzi'.

    Parameters
    ----------
    segmentation_path : str
        Path to the segmentation file (supports '.dzip' or common image formats).

    Returns
    -------
    numpy.ndarray
        A 2D or 3D segmentation array, depending on the file contents.
    """
    if segmentation_path.endswith(".dzip"):
        return reconstruct_dzi(segmentation_path)
    else:
        return cv2.imread(segmentation_path)


# related to read and write
# this function reads a VisuAlign JSON and returns the slices
def load_quint_json(filename, propagate_missing_values=True):
    """
    Reads a VisuAlign JSON file (.waln or .wwrp) and extracts slice information.
    Slices may include anchoring, grid spacing, and other image metadata.

    Parameters
    ----------
    filename : str
        The path to the VisuAlign JSON file.
    apply_damage_mask : bool
        If True, retains 'grid' data in slices; if False, removes it.

    Returns
    -------
    list
        A list of slice dictionaries containing anchoring and other metadata.
    float or None
        Grid spacing if found, otherwise None.
    """
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
    if (len(slices) > 1) & propagate_missing_values:
        slices = propagate(slices)
    vafile["slices"] = slices
    return vafile



# related to read_and_write, used in write_points_to_meshview
# this function returns a dictionary of region names
def create_region_dict(points, regions):
    """
    Groups point coordinates by their region labels and
    returns a dictionary mapping each region to its 3D point list.

    Parameters
    ----------
    points : numpy.ndarray
        A (N, 3) array of 3D coordinates for all points.
    regions : numpy.ndarray
        A 1D array of integer region labels for each point.

    Returns
    -------
    dict
        Keys are unique region labels, and values are the flattened [x, y, z, ...] coordinates.
    """
    region_dict = {
        region: points[regions == region].flatten().tolist()
        for region in np.unique(regions)
    }
    return region_dict


# related to read and write: write_points
# this function writes the region dictionary to a meshview json
def write_points(points_dict, filename, info_file):
    """
    Saves a region-based point dictionary to a MeshView-compatible JSON layout.

    Each region is recorded with: index (idx), name, color components (r, g, b),
    and a count of how many points belong to that region.

    Parameters
    ----------
    points_dict : dict
        Keys are region IDs, values are flattened 3D coordinates.
    filename : str
        Destination JSON file to be written.
    info_file : pandas.DataFrame
        A table with region IDs, names, and color data (r, g, b) for each region.
    """
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
def write_hemi_points_to_meshview(points, point_names, hemi_label, filename, info_file):
    """
    Combines point data and region information into MeshView JSON files.

    If hemisphere labels are provided (1, 2), separate outputs are saved for
    left and right hemispheres, as well as one file containing all points.

    Parameters
    ----------
    points : numpy.ndarray
        2D array containing [N, 3] point coordinates.
    point_names : numpy.ndarray
        1D array of region labels corresponding to each point.
    hemi_label : numpy.ndarray
        1D array with hemisphere labels (1 for left, 2 for right), or None.
    filename : str
        Base path for output JSON. Separate hemispheres use prefixed filenames.
    info_file : pandas.DataFrame
        A table with region IDs, names, and color data (r, g, b) for each region.
    """
    if not (hemi_label == None).all():
        split_fn_left = filename.split("/")
        split_fn_left[-1] = "left_hemisphere_" + split_fn_left[-1]
        outname_left = os.sep.join(split_fn_left)
        write_points_to_meshview(points[hemi_label == 1], point_names[hemi_label == 1], outname_left, info_file)
        split_fn_right = filename.split("/")
        split_fn_right[-1] = "right_hemisphere_" + split_fn_right[-1]
        outname_right = os.sep.join(split_fn_right)
        write_points_to_meshview(points[hemi_label == 2], point_names[hemi_label == 2], outname_right, info_file)
    write_points_to_meshview(points, point_names, filename, info_file)

# related to read and write: write_points_to_meshview
# this function combines create_region_dict and write_points functions
def write_points_to_meshview(points, point_ids, filename, info_file):
    """
    Combines point data and region information into MeshView JSON files.

    Parameters
    ----------
    points : numpy.ndarray
        2D array containing [N, 3] point coordinates.
    point_ids : numpy.ndarray
        1D array of region labels corresponding to each point.
    filename : str
        Base path for output JSON. Separate hemispheres use prefixed filenames.
    info_file : pandas.DataFrame or string
        A table with region IDs, names, and color data (r, g, b) for each region.
        If string, this should correspond to the relevant brainglobe atlas
    """
    if isinstance(info_file, str):
        info_file = load_atlas_labels(info_file)
    region_dict = create_region_dict(points, point_ids)
    write_points(region_dict, filename, info_file)
