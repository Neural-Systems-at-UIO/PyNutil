import numpy as np
import pandas as pd
import struct
import cv2
from .generate_target_slice import generate_target_slice
from .visualign_deformations import transform_vec


# related to counting and load
def label_points(points, label_volume, scale_factor=1):
    """
    Assigns points to regions based on the label_volume.

    Args:
        points (list): List of points.
        label_volume (ndarray): Volume with region labels.
        scale_factor (int, optional): Scaling factor for points. Defaults to 1.

    Returns:
        ndarray: Labels for each point.
    """
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
    """
    Counts the number of pixels per region and writes to a DataFrame.

    Args:
        labels_dict_points (dict): Dictionary with region as key and points as value.
        labeled_dict_centroids (dict): Dictionary with region as key and centroids as value.
        df_label_colours (DataFrame): DataFrame with label colours.

    Returns:
        DataFrame: DataFrame with counts and colours per region.
    """
    counted_labels_points, label_counts_points = np.unique(
        labels_dict_points, return_counts=True
    )
    counted_labels_centroids, label_counts_centroids = np.unique(
        labeled_dict_centroids, return_counts=True
    )
    # Which regions have pixels, and how many pixels are there per region
    dummy_column = np.zeros(len(counted_labels_points))
    counts_per_label = list(
        zip(counted_labels_points, label_counts_points, dummy_column)
    )
    # Create a list of unique regions and pixel counts per region
    df_counts_per_label = pd.DataFrame(
        counts_per_label, columns=["idx", "pixel_count", "object_count"]
    )
    for clc, lcc in zip(counted_labels_centroids, label_counts_centroids):
        df_counts_per_label.loc[df_counts_per_label['idx'] == clc, 'object_count'] = lcc
    new_rows = []
    for index, row in df_counts_per_label.iterrows():
        mask = df_label_colours["idx"] == row["idx"]
        current_region_row = df_label_colours[mask]
        current_region_name = current_region_row["name"].values
        current_region_red = current_region_row["r"].values
        current_region_green = current_region_row["g"].values
        current_region_blue = current_region_row["b"].values
        row["name"] = current_region_name[0]
        row["r"] = int(current_region_red[0])
        row["g"] = int(current_region_green[0])
        row["b"] = int(current_region_blue[0])

        new_rows.append(row)

    df_counts_per_label_name = pd.DataFrame(
        new_rows, columns=["idx", "name", "pixel_count", "object_count", "r", "g", "b"]
    )
    # Task for Sharon:
    # If you can get the areas per region from the flat file here
    # you can then use those areas to calculate the load per region here
    # and add to dataframe
    # see messing around pyflat.py

    return df_counts_per_label_name


"""Read flat file and write into an np array"""
"""Read flat file, write into an np array, assign label file values, return array"""


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


def rescale_image(image, rescaleXY):
    """
    Rescales an image.

    Args:
        image (ndarray): Image array.
        rescaleXY (tuple): Tuple with new dimensions.

    Returns:
        ndarray: Rescaled image.
    """
    w, h = rescaleXY
    return cv2.resize(image, (h, w), interpolation=cv2.INTER_NEAREST)


def assign_labels_to_image(image, labelfile):
    """
    Assigns labels to an image based on a label file.

    Args:
        image (ndarray): Image array.
        labelfile (DataFrame): DataFrame with label information.

    Returns:
        ndarray: Image with assigned labels.
    """
    w, h = image.shape
    allen_id_image = np.zeros((h, w))  # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))

    values = image[coordsy, coordsx]
    lbidx = labelfile["idx"].values

    allen_id_image = lbidx[values.astype(int)]
    return allen_id_image


def count_pixels_per_label(image, scale_factor=False):
    """
    Counts the number of pixels per label in an image.

    Args:
        image (ndarray): Image array.
        scale_factor (bool, optional): Whether to apply a scaling factor. Defaults to False.

    Returns:
        DataFrame: DataFrame with pixel counts per label.
    """
    unique_ids, counts = np.unique(image, return_counts=True)
    if scale_factor:
        counts = counts * scale_factor
    area_per_label = list(zip(unique_ids, counts))
    df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "region_area"])
    return df_area_per_label


def warp_image(image, triangulation, rescaleXY):
    """
    Warps an image based on triangulation.

    Args:
        image (ndarray): Image array.
        triangulation (ndarray): Triangulation data.
        rescaleXY (tuple, optional): Tuple with new dimensions. Defaults to None.

    Returns:
        ndarray: Warped image.
    """
    if rescaleXY is not None:
        w, h = rescaleXY
    else:
        h, w = image.shape
    reg_h, reg_w = image.shape
    oldX, oldY = np.meshgrid(np.arange(reg_w), np.arange(reg_h))
    oldX = oldX.flatten()
    oldY = oldY.flatten()
    h_scale = h / reg_h
    w_scale = w / reg_w
    oldX = oldX * w_scale
    oldY = oldY * h_scale
    newX, newY = transform_vec(triangulation, oldX, oldY)
    newX = newX / w_scale
    newY = newY / h_scale
    newX = newX.reshape(reg_h, reg_w)
    newY = newY.reshape(reg_h, reg_w)
    newX = newX.astype(int)
    newY = newY.astype(int)
    tempX = newX.copy()
    tempY = newY.copy()
    tempX[tempX >= reg_w] = reg_w - 1
    tempY[tempY >= reg_h] = reg_h - 1
    tempX[tempX < 0] = 0
    tempY[tempY < 0] = 0
    new_image = image[tempY, tempX]
    new_image[newX >= reg_w] = 0
    new_image[newY >= reg_h] = 0
    new_image[newX < 0] = 0
    new_image[newY < 0] = 0
    return new_image


def flat_to_dataframe(
    labelfile,
    file=None,
    rescaleXY=None,
    image_vector=None,
    volume=None,
    triangulation=None,
):
    """
    Converts a flat file to a DataFrame.

    Args:
        labelfile (DataFrame): DataFrame with label information.
        file (str, optional): Path to the flat file. Defaults to None.
        rescaleXY (tuple, optional): Tuple with new dimensions. Defaults to None.
        image_vector (ndarray, optional): Image vector. Defaults to None.
        volume (ndarray, optional): Volume data. Defaults to None.
        triangulation (ndarray, optional): Triangulation data. Defaults to None.

    Returns:
        DataFrame: DataFrame with area per label.
    """
    image = load_image(file, image_vector, volume, triangulation, rescaleXY)
    scale_factor = calculate_scale_factor(image, rescaleXY)
    allen_id_image = (
        assign_labels_to_image(image, labelfile)
        if (image_vector is None or volume is None)
        else image
    )
    df_area_per_label = count_pixels_per_label(allen_id_image, scale_factor)
    return df_area_per_label


def load_image(file, image_vector, volume, triangulation, rescaleXY):
    """
    Loads an image from a file or generates it from a vector and volume.

    Args:
        file (str): Path to the file.
        image_vector (ndarray): Image vector.
        volume (ndarray): Volume data.
        triangulation (ndarray): Triangulation data.
        rescaleXY (tuple): Tuple with new dimensions.

    Returns:
        ndarray: Loaded or generated image.
    """
    if image_vector is not None and volume is not None:
        image = generate_target_slice(image_vector, volume)
        image = np.float64(image)
        if triangulation is not None:
            image = warp_image(image, triangulation, rescaleXY)
    elif file.endswith(".flat"):
        image = read_flat_file(file)
    elif file.endswith(".seg"):
        image = read_seg_file(file)
    return image


def calculate_scale_factor(image, rescaleXY):
    """
    Calculates the scale factor for an image.

    Args:
        image (ndarray): Image array.
        rescaleXY (tuple): Tuple with new dimensions.

    Returns:
        float: Scale factor.
    """
    if rescaleXY:
        image_shapeY, image_shapeX = image.shape[0], image.shape[1]
        image_pixels = image_shapeY * image_shapeX
        seg_pixels = rescaleXY[0] * rescaleXY[1]
        return seg_pixels / image_pixels
    return False
