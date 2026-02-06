"""File loaders for PyNutil.

This module contains functions for reading various file formats:
- Custom region files (TSV, XLSX)
- Flat files (custom binary format)
- Seg files (SegRLEv1 format)
- Segmentation images (PNG, DZIP)
- VisuAlign JSON files
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from .propagation import propagate
from .reconstruct_dzi import reconstruct_dzi


def open_custom_region_file(path: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load a custom region file (TSV or XLSX).

    Returns both a dictionary of region information and a corresponding
    pandas DataFrame.

    The dictionary contains:
      - 'custom_ids': The unique IDs for each region.
      - 'custom_names': The region names.
      - 'rgb_values': The RGB values for each region.
      - 'subregion_ids': Lists of underlying atlas IDs for each region.

    Parameters
    ----------
    path : str
        Path to the TSV or XLSX file containing region information.

    Returns
    -------
    dict
        A dictionary with region data fields.
    pd.DataFrame
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

    result_df = pd.DataFrame({
        "idx": custom_region_dict["custom_ids"],
        "name": custom_region_dict["custom_names"],
        "r": [c[0] for c in custom_region_dict["rgb_values"]],
        "g": [c[1] for c in custom_region_dict["rgb_values"]],
        "b": [c[2] for c in custom_region_dict["rgb_values"]],
    })

    if result_df["name"].duplicated().any():
        raise ValueError("Duplicate region names found in custom region file.")

    return custom_region_dict, result_df


def read_flat_file(file: str) -> np.ndarray:
    """Read a custom 'flat' file format.

    This format includes a header encoding bit-depth (B), width (W), and height (H).
    Pixel data follows in a sequence that is unpacked into a 2D array.

    Parameters
    ----------
    file : str
        Path to the flat file to read.

    Returns
    -------
    np.ndarray
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


def read_seg_file(file: str) -> np.ndarray:
    """Read a segmentation file encoded with SegRLEv1 format.

    Decodes the compressed format and returns a 2D NumPy array
    representing segment labels.

    Parameters
    ----------
    file : str
        Path to the segmentation file.

    Returns
    -------
    np.ndarray
        A 2D NumPy array with segmentation labels.
    """
    with open(file, "rb") as f:

        def byte():
            return f.read(1)[0]

        def code():
            c = byte()
            if c < 0:
                raise RuntimeError("Invalid code")
            return c if c < 128 else (c & 127) | (code() << 7)

        if "SegRLEv1" != f.read(8).decode():
            raise RuntimeError("Header mismatch")

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


def load_segmentation(segmentation_path: str) -> np.ndarray:
    """Load segmentation data from a file.

    Supports '.dzip' files (processed as DZI) and standard image formats.

    Parameters
    ----------
    segmentation_path : str
        Path to the segmentation file.

    Returns
    -------
    np.ndarray
        A 2D or 3D segmentation array.
    """
    if segmentation_path.endswith(".dzip"):
        return reconstruct_dzi(segmentation_path)
    else:
        return cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)


def load_quint_json(
    filename: str,
    propagate_missing_values: bool = True,
) -> Dict[str, Any]:
    """Read a VisuAlign JSON file (.waln or .wwrp).

    Extracts slice information including anchoring, grid spacing,
    and other image metadata.

    Parameters
    ----------
    filename : str
        Path to the VisuAlign JSON file.
    propagate_missing_values : bool, optional
        If True, propagates anchoring values to slices without them.

    Returns
    -------
    dict
        Parsed VisuAlign data with 'slices' key containing slice list.
    """
    with open(filename) as f:
        vafile = json.load(f)

    if filename.endswith(".waln") or filename.endswith("wwrp"):
        slices = vafile["sections"]
        vafile["slices"] = slices
        for slice_data in slices:
            slice_data["nr"] = int(re.search(r"_s(\d+)", slice_data["filename"]).group(1))
            if "ouv" in slice_data:
                slice_data["anchoring"] = slice_data["ouv"]
    else:
        slices = vafile["slices"]

    if (len(slices) > 1) and propagate_missing_values:
        slices = propagate(slices)

    vafile["slices"] = slices
    return vafile


# ---------------------------------------------------------------------------
# File-system helpers (section numbering, flat files)
# ---------------------------------------------------------------------------


def number_sections(filenames, legacy=False):
    """Extract section numbers from a list of filenames.

    Args:
        filenames (list): List of file paths.
        legacy (bool, optional): Use a legacy extraction mode if True. Defaults to False.

    Returns:
        list: List of section numbers as integers.
    """
    filenames = [os.path.basename(filename) for filename in filenames]
    section_numbers = []
    for filename in filenames:
        if not legacy:
            # Try _s### first (standard PyNutil/QUINT format)
            match = re.findall(r"\_s(\d+)", filename)
            if len(match) == 0:
                # Try _### (common alternative)
                match = re.findall(r"\_(\d+)", filename)

            if len(match) == 0:
                raise ValueError(
                    f"No section number found in filename: {filename}. Expected format like '_s001' or '_001'."
                )

            section_numbers.append(int(match[-1]))
        else:
            match = re.sub("[^0-9]", "", filename)
            section_numbers.append(int(match[-3:]))
    if len(section_numbers) == 0:
        raise ValueError("No section numbers found in filenames")
    return section_numbers


def get_flat_files(folder, use_flat=False):
    """Retrieve flat file paths from the given folder.

    Args:
        folder (str): Path to the folder containing flat files.
        use_flat (bool, optional): If True, filter only flat files.

    Returns:
        tuple: A list of flat file paths and their numeric indices.
    """
    if use_flat:
        flat_files = [
            os.path.join(folder, "flat_files", name)
            for name in os.listdir(os.path.join(folder, "flat_files"))
            if name.endswith(".flat") or name.endswith(".seg")
        ]
        print(f"Found {len(flat_files)} flat files in folder {folder}")
        flat_file_nrs = [int(number_sections([ff])[0]) for ff in flat_files]
        return flat_files, flat_file_nrs
    return [], []


def get_current_flat_file(seg_nr, flat_files, flat_file_nrs, use_flat):
    """Determine the correct flat file for a given section number.

    Args:
        seg_nr (int): Numeric index of the segmentation.
        flat_files (list): List of flat file paths.
        flat_file_nrs (list): Numeric indices for each flat file.
        use_flat (bool): If True, attempts to match flat files to segments.

    Returns:
        str or None: The matched flat file path, or None if not found or unused.
    """
    if use_flat:
        current_flat_file_index = np.where([f == seg_nr for f in flat_file_nrs])
        return flat_files[current_flat_file_index[0][0]]
    return None
