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
        return cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)


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
def _write_points(points_dict, filename, info_file, colors_dict=None):
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
    colors_dict : dict, optional
        Keys are region IDs, values are (r, g, b) tuples to override atlas colors.
    """
    meshview = []
    for name, idx in zip(points_dict.keys(), range(len(points_dict.keys()))):
        # Find region info
        region_info = info_file[info_file["idx"] == name]
        if len(region_info) == 0:
            # Skip regions not in atlas labels
            continue

        r, g, b = int(region_info["r"].values[0]), int(region_info["g"].values[0]), int(region_info["b"].values[0])
        if colors_dict is not None:
            if name in colors_dict:
                r, g, b = colors_dict[name]

        meshview.append({
            "idx": idx,
            "count": len(points_dict[name]) // 3,
            "name": str(region_info["name"].values[0]),
            "triplets": points_dict[name],
            "r": r,
            "g": g,
            "b": b,
        })

    # write meshview json
    with open(filename, "w") as f:
        json.dump(meshview, f)


# related to read and write: write_points_to_meshview
# this function combines create_region_dict and write_points functions
def write_hemi_points_to_meshview(
    points,
    point_names,
    hemi_label,
    filename,
    info_file,
    intensities=None,
    colormap="gray",
):
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
    intensities : numpy.ndarray, optional
        1D array of intensity values for each point.
    colormap : str, optional
        Colormap to use for intensity mode (default is "gray").
    """
    if points is None or point_names is None:
        return
    if hemi_label is not None and not (hemi_label == None).all():
        split_fn_left = filename.split("/")
        split_fn_left[-1] = "left_hemisphere_" + split_fn_left[-1]
        outname_left = os.sep.join(split_fn_left)
        write_points_to_meshview(
            points[hemi_label == 1],
            point_names[hemi_label == 1],
            outname_left,
            info_file,
            intensities[hemi_label == 1] if intensities is not None else None,
            colormap,
        )
        split_fn_right = filename.split("/")
        split_fn_right[-1] = "right_hemisphere_" + split_fn_right[-1]
        outname_right = os.sep.join(split_fn_right)
        write_points_to_meshview(
            points[hemi_label == 2],
            point_names[hemi_label == 2],
            outname_right,
            info_file,
            intensities[hemi_label == 2] if intensities is not None else None,
            colormap,
        )
    write_points_to_meshview(
        points, point_names, filename, info_file, intensities, colormap
    )


# related to read and write: write_points_to_meshview
# this function combines create_region_dict and write_points functions
def write_points_to_meshview(
    points, point_ids, filename, info_file, intensities=None, colormap="gray"
):
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
    intensities : numpy.ndarray, optional
        1D array of intensity values for each point.
    colormap : str, optional
        Colormap to use for intensity mode (default is "gray").
    """
    if isinstance(info_file, str):
        info_file = load_atlas_labels(info_file)

    if intensities is not None:
        # Intensity mode: group by intensity bins instead of atlas regions
        # This creates a point cloud that looks like the original image

        if colormap == "original_colours" and intensities.ndim == 2 and intensities.shape[1] == 3:
            # RGB mode: group by unique RGB values for maximum fidelity
            rgb_data = intensities.astype(np.uint8)
            unique_colors, inverse_indices = np.unique(rgb_data, axis=0, return_inverse=True)

            # If there are too many unique colors, MeshView UI becomes slow.
            # Rounding to nearest 8 (32 levels per channel) keeps it responsive.
            if len(unique_colors) > 1024:
                rgb_data = (np.round(rgb_data / 8) * 8)
                rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
                unique_colors, inverse_indices = np.unique(rgb_data, axis=0, return_inverse=True)

            meshview = []
            for i, color in enumerate(unique_colors):
                mask = inverse_indices == i
                bin_points = points[mask]
                if len(bin_points) > 0:
                    r, g, b = color
                    meshview.append({
                        "idx": i,
                        "count": len(bin_points),
                        "name": f"Color {r},{g},{b}",
                        "triplets": bin_points.flatten().tolist(),
                        "r": int(r),
                        "g": int(g),
                        "b": int(b),
                    })
            with open(filename, "w") as f:
                json.dump(meshview, f)
            return

        # Grayscale or Colormap mode
        if intensities.ndim == 2 and intensities.shape[1] == 3:
            # Convert RGB to grayscale for binning
            intensities = (
                0.2989 * intensities[:, 0] + 0.5870 * intensities[:, 1] + 0.1140 * intensities[:, 2]
            ).astype(int)
        else:
            intensities = intensities.astype(int)

        unique_intensities = np.unique(intensities)

        meshview = []
        for val in unique_intensities:
            mask = intensities == val
            bin_points = points[mask]

            if len(bin_points) > 0:
                # Map intensity to color based on colormap
                r, g, b = _get_colormap_color(val, colormap)
                name = f"Intensity {val}"

                meshview.append(
                    {
                        "idx": int(val),
                        "count": len(bin_points),
                        "name": name,
                        "triplets": bin_points.flatten().tolist(),
                        "r": int(r),
                        "g": int(g),
                        "b": int(b),
                    }
                )

        with open(filename, "w") as f:
            json.dump(meshview, f)
        return

    region_dict = create_region_dict(points, point_ids)
    _write_points(region_dict, filename, info_file)


def _get_colormap_color(value, name="gray"):
    """
    Returns (r, g, b) for a given intensity value (0-255) and colormap name.
    """
    value = np.clip(value, 0, 255) / 255.0

    if name == "gray":
        v = int(value * 255)
        return v, v, v

    # Simple implementations of some common colormaps
    # If matplotlib is available, we could use it, but this avoids the dependency
    if name == "viridis":
        # Simplified viridis approximation
        r = 1.0 - value
        g = value
        b = 0.5 + 0.5 * value
    elif name == "plasma":
        r = value
        g = 1.0 - value
        b = 1.0 - 0.5 * value
    elif name == "magma":
        r = value
        g = value**2
        b = 1.0 - value
    elif name == "hot":
        r = min(1.0, value * 3)
        g = min(1.0, max(0.0, value * 3 - 1))
        b = min(1.0, max(0.0, value * 3 - 2))
    else:
        # Default to gray
        v = int(value * 255)
        return v, v, v

    return int(r * 255), int(g * 255), int(b * 255)
