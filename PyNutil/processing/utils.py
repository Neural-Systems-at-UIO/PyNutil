import numpy as np
import pandas as pd
import re
from glob import glob


def number_sections(filenames, legacy=False):
    """
    Extract section numbers from a list of filenames.

    Args:
        filenames (list): List of file paths.
        legacy (bool, optional): Use a legacy extraction mode if True. Defaults to False.

    Returns:
        list: List of section numbers as integers.
    """
    filenames = [filename.split("\\")[-1] for filename in filenames]
    section_numbers = []
    for filename in filenames:
        if not legacy:
            match = re.findall(r"\_s\d+", filename)
            if len(match) == 0:
                raise ValueError(f"No section number found in filename: {filename}")
            if len(match) > 1:
                raise ValueError(
                    "Multiple section numbers found in filename, ensure only one instance of _s### is present, where ### is the section number"
                )
            section_numbers.append(int(match[-1][2:]))
        else:
            match = re.sub("[^0-9]", "", filename)
            section_numbers.append(match[-3:])
    if len(section_numbers) == 0:
        raise ValueError("No section numbers found in filenames")
    return section_numbers


def find_matching_pixels(segmentation, id):
    """
    Returns the Y and X coordinates of all the pixels in the segmentation that match the id provided.

    Args:
        segmentation (ndarray): Segmentation array.
        id (int): ID to match.

    Returns:
        tuple: Y and X coordinates of matching pixels.
    """
    mask = segmentation == id
    mask = np.all(mask, axis=2)
    id_positions = np.where(mask)
    id_y, id_x = id_positions[0], id_positions[1]
    return id_y, id_x


def scale_positions(id_y, id_x, y_scale, x_scale):
    """
    Scales the Y and X coordinates to the registration space.

    Args:
        id_y (ndarray): Y coordinates.
        id_x (ndarray): X coordinates.
        y_scale (float): Y scaling factor.
        x_scale (float): X scaling factor.

    Returns:
        tuple: Scaled Y and X coordinates.
    """
    id_y = id_y * y_scale
    id_x = id_x * x_scale
    return id_y, id_x


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


def get_segmentations(folder):
    """
    Collects segmentation file paths from the specified folder.

    Args:
        folder (str): Path to the folder containing segmentations.

    Returns:
        list: List of segmentation file paths.
    """
    segmentation_file_types = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dzip"]
    segmentations = [
        file
        for file in glob(folder + "/*")
        if any([file.endswith(type) for type in segmentation_file_types])
    ]
    if len(segmentations) == 0:
        raise ValueError(
            f"No segmentations found in folder {folder}. Make sure the folder contains a segmentations folder with segmentations."
        )
    print(f"Found {len(segmentations)} segmentations in folder {folder}")
    return segmentations


def get_flat_files(folder, use_flat=False):
    """
    Retrieves flat file paths from the given folder.

    Args:
        folder (str): Path to the folder containing flat files.
        use_flat (bool, optional): If True, filter only flat files.

    Returns:
        tuple: A list of flat file paths and their numeric indices.
    """
    if use_flat:
        flat_files = [
            file
            for file in glob(folder + "/flat_files/*")
            if any([file.endswith(".flat"), file.endswith(".seg")])
        ]
        print(f"Found {len(flat_files)} flat files in folder {folder}")
        flat_file_nrs = [int(number_sections([ff])[0]) for ff in flat_files]
        return flat_files, flat_file_nrs
    return [], []


def get_current_flat_file(seg_nr, flat_files, flat_file_nrs, use_flat):
    """
    Determines the correct flat file for a given section number.

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


def start_and_join_threads(threads):
    """
    Starts a list of threads and joins them to ensure completion.

    Args:
        threads (list): A list of threading.Thread objects.

    Returns:
        None
    """
    [t.start() for t in threads]
    [t.join() for t in threads]


def process_results(
    points_list,
    centroids_list,
    points_labels,
    centroids_labels,
    points_hemi_labels,
    centroids_hemi_labels,
    points_undamaged_list,
    centroids_undamaged_list,
):
    """
    Consolidates and organizes results from multiple segmentations.

    Args:
        points_list (list): A list of arrays containing point coordinates.
        centroids_list (list): A list of arrays containing centroid coordinates.
        points_labels (list): A list of arrays with labels for each point.
        centroids_labels (list): A list of arrays with labels for each centroid.
        points_hemi_labels (list): A list of arrays storing hemisphere info per point.
        centroids_hemi_labels (list): A list of arrays storing hemisphere info per centroid.
        points_undamaged_list (list): Tracks undamaged status of each point.
        centroids_undamaged_list (list): Tracks undamaged status of each centroid.

    Returns:
        points (ndarray): Consolidated point coordinates.
        centroids (ndarray): Consolidated centroid coordinates.
        points_labels (ndarray): Combined labels for points.
        centroids_labels (ndarray): Combined labels for centroids.
        points_hemi_labels (ndarray): Combined hemisphere info for points.
        centroids_hemi_labels (ndarray): Combined hemisphere info for centroids.
        points_len (int): Total number of points.
        centroids_len (int): Total number of centroids.
        points_undamaged (ndarray): Updated track of undamaged status for points.
        centroids_undamaged (ndarray): Updated track of undamaged status for centroids.
    """
    points_len = [len(points) if None not in points else 0 for points in points_list]
    centroids_len = [
        len(centroids) if None not in centroids else 0 for centroids in centroids_list
    ]
    points_list = [
        points for points in points_list if (None not in points) and (len(points) != 0)
    ]
    centroids_list = [
        centroids
        for centroids in centroids_list
        if (None not in centroids) and (len(centroids != 0))
    ]
    points_labels = [pl for pl in points_labels if (None not in pl) and len(pl) != 0]
    centroids_labels = [
        cl for cl in centroids_labels if (None not in cl) and len(cl) != 0
    ]
    points_undamaged_list = [
        pul for pul in points_undamaged_list if (None not in pul) and len(pul) != 0
    ]
    centroids_undamaged_list = [
        cul for cul in centroids_undamaged_list if (None not in cul) and len(cul) != 0
    ]

    if len(points_list) == 0:
        points = np.array([])
        points_labels = np.array([])
        points_undamaged = np.array([])
        points_hemi_labels = np.array([])
    else:
        points = np.concatenate(points_list)
        points_labels = np.concatenate(points_labels)
        points_undamaged = np.concatenate(points_undamaged_list)
        points_hemi_labels = np.concatenate(points_hemi_labels)

    if len(centroids_list) == 0:
        centroids = np.array([])
        centroids_labels = np.array([])
        centroids_undamaged = np.array([])
        centroids_hemi_labels = np.array([])
    else:
        centroids = np.concatenate(centroids_list)
        centroids_labels = np.concatenate(centroids_labels)
        centroids_undamaged = np.concatenate(centroids_undamaged_list)
        centroids_hemi_labels = np.concatenate(centroids_hemi_labels)

    return (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        points_len,
        centroids_len,
        points_undamaged,
        centroids_undamaged,
    )
