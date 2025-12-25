import numpy as np
import pandas as pd
import struct
import cv2
from .generate_target_slice import generate_target_slice
from .visualign_deformations import transform_vec


def create_base_counts_dict(with_hemisphere=False, with_damage=False):
    """
    Creates and returns a base dictionary structure for tracking counts.

    Args:
        with_hemisphere (bool): If True, include hemisphere fields.
        with_damage (bool): If True, include damage fields.

    Returns:
        dict: Structure containing count lists for pixels/objects.
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
            "damaged_pixel_counts": [],
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
    with_damage=False,
):
    """
    Tally object counts by region, optionally tracking damage and hemispheres.

    Args:
        labels_dict_points (dict): Maps points to region labels.
        labeled_dict_centroids (dict): Maps centroids to region labels.
        current_points_undamaged (ndarray): Undamaged-state flags for points.
        current_centroids_undamaged (ndarray): Undamaged-state flags for centroids.
        current_points_hemi (ndarray): Hemisphere tags for points.
        current_centroids_hemi (ndarray): Hemisphere tags for centroids.
        df_label_colours (DataFrame): Region label colors.
        with_damage (bool, optional): Track damage counts if True.

    Returns:
        DataFrame: Summed counts per region.
    """
    # If hemisphere labels are present, they are integers (1/2). If absent, they are None.
    with_hemi = None not in current_points_hemi

    def _series_from_unique(labels: np.ndarray, counts: np.ndarray) -> pd.Series:
        if labels.size == 0:
            return pd.Series(dtype=np.int64)
        return pd.Series(counts.astype(np.int64), index=labels)

    def _counts_for(mask: np.ndarray, arr: np.ndarray) -> tuple[pd.Series, np.ndarray]:
        if arr.size == 0:
            return pd.Series(dtype=np.int64), np.array([], dtype=arr.dtype)
        if mask is None:
            u, c = np.unique(arr, return_counts=True)
        else:
            u, c = np.unique(arr[mask], return_counts=True)
        return _series_from_unique(u, c), u

    # Compute all the count series we need, then build a sparse dataframe by
    # selecting only regions with any non-zero count.
    if with_hemi and with_damage:
        lh_pu, lh_pu_idx = _counts_for(
            current_points_undamaged & (current_points_hemi == 1), labels_dict_points
        )
        lh_pd, lh_pd_idx = _counts_for(
            (~current_points_undamaged) & (current_points_hemi == 1), labels_dict_points
        )
        rh_pu, rh_pu_idx = _counts_for(
            current_points_undamaged & (current_points_hemi == 2), labels_dict_points
        )
        rh_pd, rh_pd_idx = _counts_for(
            (~current_points_undamaged) & (current_points_hemi == 2), labels_dict_points
        )

        lh_cu, lh_cu_idx = _counts_for(
            current_centroids_undamaged & (current_centroids_hemi == 1),
            labeled_dict_centroids,
        )
        lh_cd, lh_cd_idx = _counts_for(
            (~current_centroids_undamaged) & (current_centroids_hemi == 1),
            labeled_dict_centroids,
        )
        rh_cu, rh_cu_idx = _counts_for(
            current_centroids_undamaged & (current_centroids_hemi == 2),
            labeled_dict_centroids,
        )
        rh_cd, rh_cd_idx = _counts_for(
            (~current_centroids_undamaged) & (current_centroids_hemi == 2),
            labeled_dict_centroids,
        )

        all_idx = np.unique(
            np.concatenate(
                [
                    lh_pu_idx,
                    lh_pd_idx,
                    rh_pu_idx,
                    rh_pd_idx,
                    lh_cu_idx,
                    lh_cd_idx,
                    rh_cu_idx,
                    rh_cd_idx,
                ]
            )
        )
        if all_idx.size == 0:
            return pd.DataFrame(
                columns=list(
                    create_base_counts_dict(
                        with_hemisphere=True, with_damage=True
                    ).keys()
                )
            )

        base = df_label_colours[df_label_colours["idx"].isin(all_idx)].copy()
        idx = base["idx"].to_numpy()

        l_pu = lh_pu.reindex(idx, fill_value=0).to_numpy()
        l_pd = lh_pd.reindex(idx, fill_value=0).to_numpy()
        r_pu = rh_pu.reindex(idx, fill_value=0).to_numpy()
        r_pd = rh_pd.reindex(idx, fill_value=0).to_numpy()
        l_cu = lh_cu.reindex(idx, fill_value=0).to_numpy()
        l_cd = lh_cd.reindex(idx, fill_value=0).to_numpy()
        r_cu = rh_cu.reindex(idx, fill_value=0).to_numpy()
        r_cd = rh_cd.reindex(idx, fill_value=0).to_numpy()

        base["pixel_count"] = l_pu + l_pd + r_pu + r_pd
        base["undamaged_pixel_count"] = l_pu + r_pu
        base["damaged_pixel_counts"] = l_pd + r_pd
        base["object_count"] = l_cu + l_cd + r_cu + r_cd
        base["undamaged_object_count"] = l_cu + r_cu
        base["damaged_object_count"] = l_cd + r_cd

        base["left_hemi_pixel_count"] = l_pu + l_pd
        base["left_hemi_undamaged_pixel_count"] = l_pu
        base["left_hemi_damaged_pixel_count"] = l_pd
        base["left_hemi_object_count"] = l_cu + l_cd
        base["left_hemi_undamaged_object_count"] = l_cu
        base["left_hemi_damaged_object_count"] = l_cd

        base["right_hemi_pixel_count"] = r_pu + r_pd
        base["right_hemi_undamaged_pixel_count"] = r_pu
        base["right_hemi_damaged_pixel_count"] = r_pd
        base["right_hemi_object_count"] = r_cu + r_cd
        base["right_hemi_undamaged_object_count"] = r_cu
        base["right_hemi_damaged_object_count"] = r_cd

        # Keep existing naming convention for damaged pixel counts column
        # (already set as damaged_pixel_counts above).
        return base

    if with_damage and (not with_hemi):
        pu, pu_idx = _counts_for(current_points_undamaged, labels_dict_points)
        pdmg, pdmg_idx = _counts_for(~current_points_undamaged, labels_dict_points)
        cu, cu_idx = _counts_for(current_centroids_undamaged, labeled_dict_centroids)
        cd, cd_idx = _counts_for(~current_centroids_undamaged, labeled_dict_centroids)

        all_idx = np.unique(np.concatenate([pu_idx, pdmg_idx, cu_idx, cd_idx]))
        if all_idx.size == 0:
            return pd.DataFrame(
                columns=list(
                    create_base_counts_dict(
                        with_hemisphere=False, with_damage=True
                    ).keys()
                )
            )

        base = df_label_colours[df_label_colours["idx"].isin(all_idx)].copy()
        idx = base["idx"].to_numpy()

        p_u = pu.reindex(idx, fill_value=0).to_numpy()
        p_d = pdmg.reindex(idx, fill_value=0).to_numpy()
        c_u = cu.reindex(idx, fill_value=0).to_numpy()
        c_d = cd.reindex(idx, fill_value=0).to_numpy()

        base["pixel_count"] = p_u + p_d
        base["undamaged_pixel_count"] = p_u
        base["damaged_pixel_counts"] = p_d
        base["object_count"] = c_u + c_d
        base["undamaged_object_count"] = c_u
        base["damaged_object_count"] = c_d
        return base

    if with_hemi and (not with_damage):
        lh_p, lh_p_idx = _counts_for(current_points_hemi == 1, labels_dict_points)
        rh_p, rh_p_idx = _counts_for(current_points_hemi == 2, labels_dict_points)
        lh_c, lh_c_idx = _counts_for(
            current_centroids_hemi == 1, labeled_dict_centroids
        )
        rh_c, rh_c_idx = _counts_for(
            current_centroids_hemi == 2, labeled_dict_centroids
        )

        all_idx = np.unique(np.concatenate([lh_p_idx, rh_p_idx, lh_c_idx, rh_c_idx]))
        if all_idx.size == 0:
            return pd.DataFrame(
                columns=list(
                    create_base_counts_dict(
                        with_hemisphere=True, with_damage=False
                    ).keys()
                )
            )

        base = df_label_colours[df_label_colours["idx"].isin(all_idx)].copy()
        idx = base["idx"].to_numpy()

        l_p = lh_p.reindex(idx, fill_value=0).to_numpy()
        r_p = rh_p.reindex(idx, fill_value=0).to_numpy()
        l_c = lh_c.reindex(idx, fill_value=0).to_numpy()
        r_c = rh_c.reindex(idx, fill_value=0).to_numpy()

        base["pixel_count"] = l_p + r_p
        base["object_count"] = l_c + r_c
        base["left_hemi_pixel_count"] = l_p
        base["right_hemi_pixel_count"] = r_p
        base["left_hemi_object_count"] = l_c
        base["right_hemi_object_count"] = r_c
        return base

    # No damage, no hemisphere
    p, p_idx = _counts_for(None, labels_dict_points)
    c, c_idx = _counts_for(None, labeled_dict_centroids)
    all_idx = np.unique(np.concatenate([p_idx, c_idx]))
    if all_idx.size == 0:
        return pd.DataFrame(
            columns=list(
                create_base_counts_dict(with_hemisphere=False, with_damage=False).keys()
            )
        )

    base = df_label_colours[df_label_colours["idx"].isin(all_idx)].copy()
    idx = base["idx"].to_numpy()
    base["pixel_count"] = p.reindex(idx, fill_value=0).to_numpy()
    base["object_count"] = c.reindex(idx, fill_value=0).to_numpy()
    return base


def read_flat_file(file):
    """
    Reads a flat file and produces an image array.

    Args:
        file (str): Path to the flat file.

    Returns:
        ndarray: Image array extracted from the file.
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
    Reads a segmentation file into an image array.

    Args:
        file (str): Path to the segmentation file.

    Returns:
        ndarray: The segmentation image.
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
    Rescales an image to the specified dimensions.

    Args:
        image (ndarray): Input image array.
        rescaleXY (tuple): (width, height) as new size.

    Returns:
        ndarray: The rescaled image.
    """
    w, h = rescaleXY
    return cv2.resize(image, (h, w), interpolation=cv2.INTER_NEAREST)


def assign_labels_to_image(image, labelfile):
    """
    Assigns atlas or region labels to an image array.

    Args:
        image (ndarray): Image array to label.
        labelfile (DataFrame): Contains label IDs in the 'idx' column.

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
    Counts the pixels associated with each label in an image.

    Args:
        image (ndarray): Image array containing labels.
        scale_factor (bool, optional): Apply scaling if True.

    Returns:
        DataFrame: Table of label IDs and pixel counts.
    """
    unique_ids, counts = np.unique(image, return_counts=True)
    if scale_factor:
        counts = counts * scale_factor
    area_per_label = list(zip(unique_ids, counts))
    df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "region_area"])
    return df_area_per_label


def warp_image(image, triangulation, rescaleXY):
    """
    Warps an image using triangulation, applying optional resizing.

    Args:
        image (ndarray): Image array to be warped.
        triangulation (ndarray): Triangulation data for remapping.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        ndarray: The warped image array.
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
    Builds a DataFrame from an image, incorporating optional damage/hemisphere masks.

    Args:
        image (ndarray): Source image with label IDs.
        damage_mask (ndarray): Binary mask indicating damaged areas.
        hemi_mask (ndarray): Binary mask for hemisphere assignment.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        DataFrame: Pixel counts grouped by label.
        ndarray: Scaled label map of the image.
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
        combos = [
            (None, None, "region_area")
        ]  # compute for entire image with no filtering

    # Count pixels for each combo
    for hemi_val, damage_val, col_name in combos:
        mask = np.ones_like(image, dtype=bool)
        if hemi_mask is not None:
            mask &= hemi_mask == hemi_val
        if damage_mask is not None:
            mask &= damage_mask == damage_val
        combo_df = count_pixels_per_label(image[mask], scale_factor)
        combo_df = combo_df.rename(columns={"region_area": col_name})
        df_area_per_label = pd.merge(
            df_area_per_label, combo_df, on="idx", how="outer"
        ).fillna(0)

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
    Loads an image from file or transforms a preloaded array, optionally applying warping.

    Args:
        file (str): File path for the source image.
        image_vector (ndarray): Preloaded image data array.
        volume (ndarray): Atlas volume or similar data.
        triangulation (ndarray): Triangulation data for warping.
        rescaleXY (tuple): (width, height) for resizing.
        labelfile (DataFrame, optional): Label definitions.

    Returns:
        ndarray: The loaded or transformed image.
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
    Computes a factor for resizing if needed.

    Args:
        image (ndarray): Original image array.
        rescaleXY (tuple): (width, height) for potential resizing.

    Returns:
        float or bool: Scale factor or False if not applicable.
    """
    if rescaleXY:
        image_shapeY, image_shapeX = image.shape[0], image.shape[1]
        image_pixels = image_shapeY * image_shapeX
        seg_pixels = rescaleXY[0] * rescaleXY[1]
        return seg_pixels / image_pixels
    return False
