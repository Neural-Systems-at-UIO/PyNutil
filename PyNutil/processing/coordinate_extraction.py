import numpy as np
import pandas as pd
from ..io.read_and_write import load_visualign_json
from .counting_and_load import flat_to_dataframe
from .visualign_deformations import triangulate
from glob import glob
import cv2
from skimage import measure
import threading
from ..io.reconstruct_dzi import reconstruct_dzi
from .transformations import (
    transform_points_to_atlas_space,
    transform_to_registration,
    get_transformed_coordinates,
)
from .utils import (
    get_flat_files,
    get_segmentations,
    number_sections,
    find_matching_pixels,
    scale_positions,
    process_results,
    get_current_flat_file,
    start_and_join_threads,
)


def get_centroids_and_area(segmentation, pixel_cut_off=0):
    """
    Returns the center coordinate of each object in the segmentation.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_cut_off (int, optional): Pixel cutoff to remove small objects. Defaults to 0.

    Returns:
        tuple: Centroids, area, and coordinates of objects.
    """
    labels = measure.label(segmentation)
    labels_info = measure.regionprops(labels)
    labels_info = [label for label in labels_info if label.area > pixel_cut_off]
    centroids = np.array([label.centroid for label in labels_info])
    area = np.array([label.area for label in labels_info])
    coords = np.array([label.coords for label in labels_info], dtype=object)
    return centroids, area, coords



def create_threads(
    segmentations,
    slices,
    flat_files,
    flat_file_nrs,
    atlas_labels,
    pixel_id,
    non_linear,
    points_list,
    centroids_list,
    region_areas_list,
    object_cutoff,
    atlas_volume,
    use_flat,
):
    """
    Creates threads for processing segmentations.

    Args:
        segmentations (list): List of segmentation files.
        slices (list): List of slices.
        flat_files (list): List of flat files.
        flat_file_nrs (list): List of flat file section numbers.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        pixel_id (list): Pixel ID to match.
        non_linear (bool): Whether to use non-linear transformation.
        points_list (list): List to store points.
        centroids_list (list): List to store centroids.
        region_areas_list (list): List to store region areas.
        object_cutoff (int): Pixel cutoff to remove small objects.
        atlas_volume (ndarray): Volume with atlas labels.
        use_flat (bool): Whether to use flat files.

    Returns:
        list: List of threads.
    """
    threads = []
    for segmentation_path, index in zip(segmentations, range(len(segmentations))):
        seg_nr = int(number_sections([segmentation_path])[0])
        current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
        current_slice = slices[current_slice_index[0][0]]
        if current_slice["anchoring"] == []:
            continue
        current_flat = get_current_flat_file(
            seg_nr, flat_files, flat_file_nrs, use_flat
        )

        x = threading.Thread(
            target=segmentation_to_atlas_space,
            args=(
                current_slice,
                segmentation_path,
                atlas_labels,
                current_flat,
                pixel_id,
                non_linear,
                points_list,
                centroids_list,
                region_areas_list,
                index,
                object_cutoff,
                atlas_volume,
                use_flat,
            ),
        )
        threads.append(x)
    return threads


def folder_to_atlas_space(
    folder,
    quint_alignment,
    atlas_labels,
    pixel_id=[0, 0, 0],
    non_linear=True,
    object_cutoff=0,
    atlas_volume=None,
    use_flat=False,
):
    """
    Applies segmentation to atlas space for all segmentations in a folder.

    Args:
        folder (str): Path to the folder.
        quint_alignment (str): Path to the QuickNII alignment file.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        pixel_id (list, optional): Pixel ID to match. Defaults to [0, 0, 0].
        non_linear (bool, optional): Whether to use non-linear transformation. Defaults to True.
        object_cutoff (int, optional): Pixel cutoff to remove small objects. Defaults to 0.
        atlas_volume (ndarray, optional): Volume with atlas labels. Defaults to None.
        use_flat (bool, optional): Whether to use flat files. Defaults to False.

    Returns:
        tuple: Points, centroids, region areas list, points length, centroids length, segmentations.
    """
    slices = load_visualign_json(quint_alignment)
    segmentations = get_segmentations(folder)
    flat_files, flat_file_nrs = get_flat_files(folder, use_flat)

    points_list, centroids_list, region_areas_list = initialize_lists(
        len(segmentations)
    )
    threads = create_threads(
        segmentations,
        slices,
        flat_files,
        flat_file_nrs,
        atlas_labels,
        pixel_id,
        non_linear,
        points_list,
        centroids_list,
        region_areas_list,
        object_cutoff,
        atlas_volume,
        use_flat,
    )
    start_and_join_threads(threads)

    points, centroids, points_len, centroids_len = process_results(
        points_list, centroids_list
    )
    return (
        points,
        centroids,
        region_areas_list,
        points_len,
        centroids_len,
        segmentations,
    )





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


def create_threads(
    segmentations,
    slices,
    flat_files,
    flat_file_nrs,
    atlas_labels,
    pixel_id,
    non_linear,
    points_list,
    centroids_list,
    region_areas_list,
    object_cutoff,
    atlas_volume,
    use_flat,
):
    """
    Creates threads for processing segmentations.

    Args:
        segmentations (list): List of segmentation files.
        slices (list): List of slices.
        flat_files (list): List of flat files.
        flat_file_nrs (list): List of flat file section numbers.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        pixel_id (list): Pixel ID to match.
        non_linear (bool): Whether to use non-linear transformation.
        points_list (list): List to store points.
        centroids_list (list): List to store centroids.
        region_areas_list (list): List to store region areas.
        object_cutoff (int): Pixel cutoff to remove small objects.
        atlas_volume (ndarray): Volume with atlas labels.
        use_flat (bool): Whether to use flat files.

    Returns:
        list: List of threads.
    """
    threads = []
    for segmentation_path, index in zip(segmentations, range(len(segmentations))):
        seg_nr = int(number_sections([segmentation_path])[0])
        current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
        current_slice = slices[current_slice_index[0][0]]
        if current_slice["anchoring"] == []:
            continue
        current_flat = get_current_flat_file(
            seg_nr, flat_files, flat_file_nrs, use_flat
        )

        x = threading.Thread(
            target=segmentation_to_atlas_space,
            args=(
                current_slice,
                segmentation_path,
                atlas_labels,
                current_flat,
                pixel_id,
                non_linear,
                points_list,
                centroids_list,
                region_areas_list,
                index,
                object_cutoff,
                atlas_volume,
                use_flat,
            ),
        )
        threads.append(x)
    return threads


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


def detect_pixel_id(segmentation: np.array):
    """
    Removes the background from the segmentation and returns the pixel ID.

    Args:
        segmentation (ndarray): Segmentation array.

    Returns:
        ndarray: Pixel ID.
    """
    segmentation_no_background = segmentation[~np.all(segmentation == 0, axis=2)]
    pixel_id = segmentation_no_background[0]
    print("detected pixel_id: ", pixel_id)
    return pixel_id


def get_region_areas(
    use_flat,
    atlas_labels,
    flat_file_atlas,
    seg_width,
    seg_height,
    slice_dict,
    atlas_volume,
    triangulation,
):
    """
    Gets the region areas.

    Args:
        use_flat (bool): Whether to use flat files.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        flat_file_atlas (str): Path to the flat file atlas.
        seg_width (int): Segmentation width.
        seg_height (int): Segmentation height.
        slice_dict (dict): Dictionary with slice information.
        atlas_volume (ndarray): Volume with atlas labels.
        triangulation (ndarray): Triangulation data.

    Returns:
        DataFrame: DataFrame with region areas.
    """
    if use_flat:
        region_areas = flat_to_dataframe(
            atlas_labels, flat_file_atlas, (seg_width, seg_height)
        )
    else:
        region_areas = flat_to_dataframe(
            atlas_labels,
            flat_file_atlas,
            (seg_width, seg_height),
            slice_dict["anchoring"],
            atlas_volume,
            triangulation,
        )
    return region_areas


def segmentation_to_atlas_space(
    slice_dict,
    segmentation_path,
    atlas_labels,
    flat_file_atlas=None,
    pixel_id="auto",
    non_linear=True,
    points_list=None,
    centroids_list=None,
    region_areas_list=None,
    index=None,
    object_cutoff=0,
    atlas_volume=None,
    use_flat=False,
):
    """
    Converts a segmentation to atlas space.

    Args:
        slice_dict (dict): Dictionary with slice information.
        segmentation_path (str): Path to the segmentation file.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        flat_file_atlas (str, optional): Path to the flat file atlas. Defaults to None.
        pixel_id (str, optional): Pixel ID to match. Defaults to "auto".
        non_linear (bool, optional): Whether to use non-linear transformation. Defaults to True.
        points_list (list, optional): List to store points. Defaults to None.
        centroids_list (list, optional): List to store centroids. Defaults to None.
        region_areas_list (list, optional): List to store region areas. Defaults to None.
        index (int, optional): Index of the current segmentation. Defaults to None.
        object_cutoff (int, optional): Pixel cutoff to remove small objects. Defaults to 0.
        atlas_volume (ndarray, optional): Volume with atlas labels. Defaults to None.
        use_flat (bool, optional): Whether to use flat files. Defaults to False.
    """
    segmentation = load_segmentation(segmentation_path)
    if pixel_id == "auto":
        pixel_id = detect_pixel_id(segmentation)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    triangulation = get_triangulation(slice_dict, reg_width, reg_height, non_linear)
    region_areas = get_region_areas(
        use_flat,
        atlas_labels,
        flat_file_atlas,
        seg_width,
        seg_height,
        slice_dict,
        atlas_volume,
        triangulation,
    )
    y_scale, x_scale = transform_to_registration(
        seg_height, seg_width, reg_height, reg_width
    )
    centroids, points = None, None
    scaled_centroidsX, scaled_centroidsY, scaled_x, scaled_y = None, None, None, None
    centroids, scaled_centroidsX, scaled_centroidsY = get_centroids(
        segmentation, pixel_id, y_scale, x_scale, object_cutoff
    )
    scaled_y, scaled_x = get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale)

    new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(
        non_linear,
        slice_dict,
        scaled_x,
        scaled_y,
        centroids,
        scaled_centroidsX,
        scaled_centroidsY,
        triangulation,
    )
    points, centroids = transform_points_to_atlas_space(
        slice_dict,
        new_x,
        new_y,
        centroids_new_x,
        centroids_new_y,
        reg_height,
        reg_width,
    )
    points_list[index] = np.array(points if points is not None else [])
    centroids_list[index] = np.array(centroids if centroids is not None else [])
    region_areas_list[index] = region_areas


def get_triangulation(slice_dict, reg_width, reg_height, non_linear):
    """
    Gets the triangulation for the slice.

    Args:
        slice_dict (dict): Dictionary with slice information.
        reg_width (int): Registration width.
        reg_height (int): Registration height.
        non_linear (bool): Whether to use non-linear transformation.

    Returns:
        list: Triangulation data.
    """
    if non_linear and "markers" in slice_dict:
        return triangulate(reg_width, reg_height, slice_dict["markers"])
    return None


def get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff=0):
    """
    Gets the centroids of objects in the segmentation.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel ID to match.
        y_scale (float): Y scaling factor.
        x_scale (float): X scaling factor.
        object_cutoff (int, optional): Pixel cutoff to remove small objects. Defaults to 0.

    Returns:
        tuple: Centroids, scaled X coordinates, and scaled Y coordinates.
    """
    binary_seg = segmentation == pixel_id
    binary_seg = np.all(binary_seg, axis=2)
    centroids, area, coords = get_centroids_and_area(
        binary_seg, pixel_cut_off=object_cutoff
    )
    if len(centroids) == 0:
        return None, None, None
    centroidsX = centroids[:, 1]
    centroidsY = centroids[:, 0]
    scaled_centroidsY, scaled_centroidsX = scale_positions(
        centroidsY, centroidsX, y_scale, x_scale
    )
    return centroids, scaled_centroidsX, scaled_centroidsY


def get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale):
    """
    Gets the scaled pixel coordinates.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel ID to match.
        y_scale (float): Y scaling factor.
        x_scale (float): X scaling factor.

    Returns:
        tuple: Scaled Y and X coordinates.
    """
    id_pixels = find_matching_pixels(segmentation, pixel_id)
    if len(id_pixels[0]) == 0:
        return None, None
    scaled_y, scaled_x = scale_positions(id_pixels[0], id_pixels[1], y_scale, x_scale)
    return scaled_y, scaled_x
