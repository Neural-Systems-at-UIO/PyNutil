import numpy as np
import pandas as pd
import re
import threading
from glob import glob


def number_sections(filenames, legacy=False):
    """
    Returns the section numbers of filenames.

    Args:
        filenames (list): List of filenames.
        legacy (bool, optional): Whether to use legacy mode. Defaults to False.

    Returns:
        list: List of section numbers.
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
    Gets the list of segmentation files in the folder.

    Args:
        folder (str): Path to the folder.

    Returns:
        list: List of segmentation files.
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


def get_flat_files(folder, use_flat):
    """
    Gets the list of flat files in the folder.

    Args:
        folder (str): Path to the folder.
        use_flat (bool): Whether to use flat files.

    Returns:
        tuple: List of flat files and their section numbers.
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


def initialize_lists(length):
    """
    Initializes lists for storing points, centroids, and region areas.

    Args:
        length (int): Length of the lists.

    Returns:
        tuple: Initialized lists.
    """
    points_list = [np.array([])] * length
    centroids_list = [np.array([])] * length
    region_areas_list = [
        pd.DataFrame(
            {
                "idx": [],
                "name": [],
                "r": [],
                "g": [],
                "b": [],
                "region_area": [],
                "pixel_count": [],
                "object_count": [],
                "area_fraction": [],
            }
        )
    ] * length
    return points_list, centroids_list, region_areas_list


def get_current_flat_file(seg_nr, flat_files, flat_file_nrs, use_flat):
    """
    Gets the current flat file for the given section number.

    Args:
        seg_nr (int): Section number.
        flat_files (list): List of flat files.
        flat_file_nrs (list): List of flat file section numbers.
        use_flat (bool): Whether to use flat files.

    Returns:
        str: Path to the current flat file.
    """
    if use_flat:
        current_flat_file_index = np.where([f == seg_nr for f in flat_file_nrs])
        return flat_files[current_flat_file_index[0][0]]
    return None


def start_and_join_threads(threads):
    """
    Starts and joins the threads.

    Args:
        threads (list): List of threads.
    """
    [t.start() for t in threads]
    [t.join() for t in threads]


def process_results(points_list, centroids_list):
    """
    Processes the results from the threads.

    Args:
        points_list (list): List of points.
        centroids_list (list): List of centroids.

    Returns:
        tuple: Processed points, centroids, points length, and centroids length.
    """
    points_len = [len(points) if None not in points else 0 for points in points_list]
    centroids_len = [
        len(centroids) if None not in centroids else 0 for centroids in centroids_list
    ]
    points_list = [points for points in points_list if None not in points]
    centroids_list = [
        centroids for centroids in centroids_list if None not in centroids
    ]
    if len(points_list) == 0:
        points = np.array([])
    else:
        points = np.concatenate(points_list)
    if len(centroids_list) == 0:
        centroids = np.array([])
    else:
        centroids = np.concatenate(centroids_list)
    return points, centroids, points_len, centroids_len
