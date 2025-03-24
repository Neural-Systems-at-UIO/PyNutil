import numpy as np
import pandas as pd
import struct
import cv2
from .generate_target_slice import generate_target_slice
from .visualign_deformations import transform_vec

def create_base_counts_dict(with_hemisphere=False, with_damage=False):
    """
    Creates the base dictionary structure for counts.

    Args:
        with_hemisphere (bool): Whether to include hemisphere-specific fields

    Returns:
        dict: Base dictionary with count fields
    """
    counts = {
        "idx": [],
        "name": [],
        "r": [],
        "g": [],
        "b": [],
        "pixel_count": [],
        "object_count": [],
    }
    if with_damage:
        damage_fields = {
            "undamaged_object_count": [],
            "damaged_object_count": [],
            "undamaged_pixel_count": [],
            "damaged_pixel_counts": []
            }
        counts.update(damage_fields)
    if with_hemisphere:
        hemisphere_fields = {
            "left_hemi_pixel_count": [],
            "right_hemi_pixel_count": [],
            "left_hemi_object_count": [],
            "right_hemi_object_count": [],
            }
        counts.update(hemisphere_fields)
    if with_damage and with_hemisphere:
        damage_hemisphere_fields = {
            "left_hemi_undamaged_pixel_count": [],
            "left_hemi_damaged_pixel_count": [],
            "right_hemi_undamaged_pixel_count": [],
            "right_hemi_damaged_pixel_count": [],
            "left_hemi_undamaged_object_count": [],
            "left_hemi_damaged_object_count": [],
            "right_hemi_undamaged_object_count": [],
            "right_hemi_damaged_object_count": [],
            }
        counts.update(damage_hemisphere_fields)
    return counts

# related to counting_and_load
def pixel_count_per_region(
    labels_dict_points,
    labeled_dict_centroids,
    current_points_undamaged,
    current_centroids_undamaged,
    current_points_hemi,
    current_centroids_hemi,
    df_label_colours,
    with_damage=False
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
    with_hemi = None not in current_points_hemi
    counts_per_label = create_base_counts_dict(with_hemisphere=with_hemi, with_damage=with_damage)

    if with_hemi and with_damage:
        left_hemi_counted_labels_points_undamaged, left_hemi_label_counts_points_undamaged = np.unique(
            labels_dict_points[current_points_undamaged & (current_points_hemi == 1)], return_counts=True
            )
        left_hemi_counted_labels_points_damaged, left_hemi_label_counts_points_damaged = np.unique(
            labels_dict_points[~current_points_undamaged & (current_points_hemi == 1)], return_counts=True
            )
        left_hemi_counted_labels_centroids_undamaged, left_hemi_label_counts_centroids_undamaged = np.unique(
            labeled_dict_centroids[current_centroids_undamaged & (current_centroids_hemi == 1)], return_counts=True
            )
        left_hemi_counted_labels_centroids_damaged, left_hemi_label_counts_centroids_damaged = np.unique(
            labeled_dict_centroids[~current_centroids_undamaged & (current_centroids_hemi == 1)], return_counts=True
            )
        right_hemi_counted_labels_points_undamaged, right_hemi_label_counts_points_undamaged = np.unique(
            labels_dict_points[current_points_undamaged & (current_points_hemi == 2)], return_counts=True
            )
        right_hemi_counted_labels_points_damaged, right_hemi_label_counts_points_damaged = np.unique(
            labels_dict_points[~current_points_undamaged & (current_points_hemi == 2)], return_counts=True
            )
        right_hemi_counted_labels_centroids_undamaged, right_hemi_label_counts_centroids_undamaged = np.unique(
            labeled_dict_centroids[current_centroids_undamaged & (current_centroids_hemi == 2)], return_counts=True
            )
        right_hemi_counted_labels_centroids_damaged, right_hemi_label_counts_centroids_damaged = np.unique(
            labeled_dict_centroids[~current_centroids_undamaged & (current_centroids_hemi == 2)], return_counts=True
            )
        for index, row in df_label_colours.iterrows():
            # Left hemisphere pixel counts
            if row["idx"] in left_hemi_counted_labels_points_undamaged:
                l_clpu = left_hemi_label_counts_points_undamaged[
                    left_hemi_counted_labels_points_undamaged == row["idx"]
                ][0]
            else:
                l_clpu = 0

            if row["idx"] in left_hemi_counted_labels_points_damaged:
                l_clpd = left_hemi_label_counts_points_damaged[
                    left_hemi_counted_labels_points_damaged == row["idx"]
                ][0]
            else:
                l_clpd = 0

            # Right hemisphere pixel counts
            if row["idx"] in right_hemi_counted_labels_points_undamaged:
                r_clpu = right_hemi_label_counts_points_undamaged[
                    right_hemi_counted_labels_points_undamaged == row["idx"]
                ][0]
            else:
                r_clpu = 0

            if row["idx"] in right_hemi_counted_labels_points_damaged:
                r_clpd = right_hemi_label_counts_points_damaged[
                    right_hemi_counted_labels_points_damaged == row["idx"]
                ][0]
            else:
                r_clpd = 0

            # Left hemisphere object counts
            if row["idx"] in left_hemi_counted_labels_centroids_undamaged:
                l_clcu = left_hemi_label_counts_centroids_undamaged[
                    left_hemi_counted_labels_centroids_undamaged == row["idx"]
                ][0]
            else:
                l_clcu = 0

            if row["idx"] in left_hemi_counted_labels_centroids_damaged:
                l_clcd = left_hemi_label_counts_centroids_damaged[
                    left_hemi_counted_labels_centroids_damaged == row["idx"]
                ][0]
            else:
                l_clcd = 0

            # Right hemisphere object counts
            if row["idx"] in right_hemi_counted_labels_centroids_undamaged:
                r_clcu = right_hemi_label_counts_centroids_undamaged[
                    right_hemi_counted_labels_centroids_undamaged == row["idx"]
                ][0]
            else:
                r_clcu = 0

            if row["idx"] in right_hemi_counted_labels_centroids_damaged:
                r_clcd = right_hemi_label_counts_centroids_damaged[
                    right_hemi_counted_labels_centroids_damaged == row["idx"]
                ][0]
            else:
                r_clcd = 0

            # Skip regions with no counts in any category
            if l_clcd == l_clcu == l_clpd == l_clpu == r_clcd == r_clcu == r_clpd == r_clpu == 0:
                continue

            # Calculate combined counts
            clpu = l_clpu + r_clpu  # total undamaged pixel count
            clpd = l_clpd + r_clpd  # total damaged pixel count
            clcu = l_clcu + r_clcu  # total undamaged object count
            clcd = l_clcd + r_clcd  # total damaged object count

            # Add to dictionary
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))

            # Total counts
            counts_per_label["pixel_count"].append(clpu + clpd)
            counts_per_label["undamaged_pixel_count"].append(clpu)
            counts_per_label["damaged_pixel_counts"].append(clpd)
            counts_per_label["object_count"].append(clcu + clcd)
            counts_per_label["undamaged_object_count"].append(clcu)
            counts_per_label["damaged_object_count"].append(clcd)

            # Left hemisphere counts
            counts_per_label["left_hemi_pixel_count"].append(l_clpu + l_clpd)
            counts_per_label["left_hemi_undamaged_pixel_count"].append(l_clpu)
            counts_per_label["left_hemi_damaged_pixel_count"].append(l_clpd)
            counts_per_label["left_hemi_object_count"].append(l_clcu + l_clcd)
            counts_per_label["left_hemi_undamaged_object_count"].append(l_clcu)
            counts_per_label["left_hemi_damaged_object_count"].append(l_clcd)

            # Right hemisphere counts
            counts_per_label["right_hemi_pixel_count"].append(r_clpu + r_clpd)
            counts_per_label["right_hemi_undamaged_pixel_count"].append(r_clpu)
            counts_per_label["right_hemi_damaged_pixel_count"].append(r_clpd)
            counts_per_label["right_hemi_object_count"].append(r_clcu + r_clcd)
            counts_per_label["right_hemi_undamaged_object_count"].append(r_clcu)
            counts_per_label["right_hemi_damaged_object_count"].append(r_clcd)

    elif with_damage and (not with_hemi):
        counted_labels_points_undamaged, label_counts_points_undamaged = np.unique(
        labels_dict_points[current_points_undamaged], return_counts=True
            )
        counted_labels_points_damaged, label_counts_points_damaged = np.unique(
            labels_dict_points[~current_points_undamaged], return_counts=True
            )
        counted_labels_centroids_undamaged, label_counts_centroids_undamaged = np.unique(
            labeled_dict_centroids[current_centroids_undamaged], return_counts=True
            )
        counted_labels_centroids_damaged, label_counts_centroids_damaged = np.unique(
            labeled_dict_centroids[~current_centroids_undamaged], return_counts=True
            )
        for index, row in df_label_colours.iterrows():
            if row["idx"] in counted_labels_points_undamaged:
                clpu = label_counts_points_undamaged[
                    counted_labels_points_undamaged == row["idx"]
                ][0]
            else:
                clpu = 0
            if row["idx"] in counted_labels_points_damaged:
                clpd = label_counts_points_damaged[
                    counted_labels_points_damaged == row["idx"]
                ][0]
            else:
                clpd = 0
            if row["idx"] in counted_labels_centroids_undamaged:
                clcu = label_counts_centroids_undamaged[
                    counted_labels_centroids_undamaged == row["idx"]
                ][0]
            else:
                clcu = 0
            if row["idx"] in counted_labels_centroids_damaged:
                clcd = label_counts_centroids_damaged[
                    counted_labels_centroids_damaged == row["idx"]
                ][0]
            else:
                clcd = 0
            if clcd == clcu == clpd == clpu == 0:
                continue
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))
            counts_per_label["pixel_count"].append(clpu + clpd)
            counts_per_label["undamaged_pixel_count"].append(clpu)
            counts_per_label["damaged_pixel_counts"].append(clpd)
            counts_per_label["object_count"].append(clcu + clcd)
            counts_per_label["undamaged_object_count"].append(clcu)
            counts_per_label["damaged_object_count"].append(clcd)

    elif with_hemi and (not with_damage):
        left_hemi_counted_labels_points, left_hemi_label_counts_points = np.unique(
            labels_dict_points[current_points_hemi == 1], return_counts=True
        )
        left_hemi_counted_labels_centroids, left_hemi_label_counts_centroids = np.unique(
            labeled_dict_centroids[current_centroids_hemi == 1], return_counts=True
        )
        right_hemi_counted_labels_points, right_hemi_label_counts_points = np.unique(
            labels_dict_points[current_points_hemi == 2], return_counts=True
        )
        right_hemi_counted_labels_centroids, right_hemi_label_counts_centroids = np.unique(
            labeled_dict_centroids[current_centroids_hemi == 2], return_counts=True
        )

        for index, row in df_label_colours.iterrows():
            # Left hemisphere
            l_clp = (
                left_hemi_label_counts_points[left_hemi_counted_labels_points == row["idx"]][0]
                if row["idx"] in left_hemi_counted_labels_points else 0
            )
            l_clc = (
                left_hemi_label_counts_centroids[left_hemi_counted_labels_centroids == row["idx"]][0]
                if row["idx"] in left_hemi_counted_labels_centroids else 0
            )
            # Right hemisphere
            r_clp = (
                right_hemi_label_counts_points[right_hemi_counted_labels_points == row["idx"]][0]
                if row["idx"] in right_hemi_counted_labels_points else 0
            )
            r_clc = (
                right_hemi_label_counts_centroids[right_hemi_counted_labels_centroids == row["idx"]][0]
                if row["idx"] in right_hemi_counted_labels_centroids else 0
            )

            # Skip empty counts
            if l_clp == r_clp == l_clc == r_clc == 0:
                continue

            # Add to dictionary
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))
            counts_per_label["pixel_count"].append(l_clp + r_clp)
            counts_per_label["object_count"].append(l_clc + r_clc)
            counts_per_label["left_hemi_pixel_count"].append(l_clp)
            counts_per_label["right_hemi_pixel_count"].append(r_clp)
            counts_per_label["left_hemi_object_count"].append(l_clc)
            counts_per_label["right_hemi_object_count"].append(r_clc)

    else:
        counted_labels_points, label_counts_points = np.unique(labels_dict_points, return_counts=True)
        counted_labels_centroids, label_counts_centroids = np.unique(labeled_dict_centroids, return_counts=True)

        for index, row in df_label_colours.iterrows():
            clp = (
                label_counts_points[counted_labels_points == row["idx"]][0]
                if row["idx"] in counted_labels_points else 0
            )
            clc = (
                label_counts_centroids[counted_labels_centroids == row["idx"]][0]
                if row["idx"] in counted_labels_centroids else 0
            )
            if clp == 0 and clc == 0:
                continue
            counts_per_label["idx"].append(row["idx"])
            counts_per_label["name"].append(row["name"])
            counts_per_label["r"].append(int(row["r"]))
            counts_per_label["g"].append(int(row["g"]))
            counts_per_label["b"].append(int(row["b"]))
            counts_per_label["pixel_count"].append(clp)
            counts_per_label["object_count"].append(clc)

    df_counts_per_label = pd.DataFrame(counts_per_label)
    return df_counts_per_label


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
        rescaleXY (tuple): Tuple with new width and height.

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


def flat_to_dataframe(image, damage_mask, hemi_mask, rescaleXY=None):
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
        np.array: array in shape of alignment XY scaled by rescaleXY with allen ID for each point
    """
    scale_factor = calculate_scale_factor(image, rescaleXY)
    df_area_per_label = pd.DataFrame(columns=["idx"])
    if hemi_mask is not None:
        hemi_mask = cv2.resize(
            hemi_mask.astype(np.uint8),
            (image.shape[::-1]),
            interpolation=cv2.INTER_NEAREST,
        )

    if damage_mask is not None:
        damage_mask = cv2.resize(
            damage_mask.astype(np.uint8),
            (image.shape[::-1]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Build combinations for each scenario
    if (hemi_mask is not None) and (damage_mask is not None):
        combos = [
            (1, 0, "left_hemi_undamaged_region_area"),
            (1, 1, "left_hemi_damaged_region_area"),
            (2, 0, "right_hemi_undamaged_region_area"),
            (2, 1, "right_hemi_damaged_region_area"),
        ]
    elif (hemi_mask is not None) and (damage_mask is None):
        combos = [
            (1, 0, "left_hemi_region_area"),
            (2, 0, "right_hemi_region_area"),
        ]
    elif (hemi_mask is None) and (damage_mask is not None):
        combos = [
            (0, 0, "undamaged_region_area"),
            (0, 1, "damaged_region_area"),
        ]
    else:
        combos = [(None, None, "region_area")]  # compute for entire image with no filtering

    # Count pixels for each combo
    for hemi_val, damage_val, col_name in combos:
        mask = np.ones_like(image, dtype=bool)
        if hemi_mask is not None:
            mask &= (hemi_mask == hemi_val)
        if damage_mask is not None:
            mask &= (damage_mask == damage_val)
        combo_df = count_pixels_per_label(image[mask], scale_factor)
        combo_df = combo_df.rename(columns={"region_area": col_name})
        df_area_per_label = pd.merge(df_area_per_label, combo_df, on="idx", how="outer").fillna(0)

    # If both masks exist, compute additional columns
    if (hemi_mask is not None) and (damage_mask is not None):
        df_area_per_label["undamaged_region_area"] = (
            df_area_per_label["left_hemi_undamaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["damaged_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_damaged_region_area"]
        )
        df_area_per_label["left_hemi_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["left_hemi_undamaged_region_area"]
        )
        df_area_per_label["right_hemi_region_area"] = (
            df_area_per_label["right_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    if (hemi_mask is not None) and (damage_mask is None):
        df_area_per_label["region_area"] = (
            df_area_per_label["left_hemi_region_area"]
            + df_area_per_label["right_hemi_region_area"]
        )
    if (hemi_mask is None) and (damage_mask is not None):
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    return df_area_per_label


def load_image(file, image_vector, volume, triangulation, rescaleXY, labelfile=None):
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
    else:
        if file.endswith(".flat"):
            image = read_flat_file(file)
        if file.endswith(".seg"):
            image = read_seg_file(file)
        image = assign_labels_to_image(image, labelfile)

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
