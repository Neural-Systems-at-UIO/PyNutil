import numpy as np
import pandas as pd
from ..io.read_and_write import load_quint_json
from .counting_and_load import flat_to_dataframe, rescale_image, load_image
from .generate_target_slice import generate_target_slice
from .visualign_deformations import triangulate
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
    Retrieves centroids, areas, and pixel coordinates of labeled regions.

    Args:
        segmentation (ndarray): Binary segmentation array.
        pixel_cut_off (int, optional): Minimum object size threshold.

    Returns:
        tuple: (centroids, area, coords) of retained objects.
    """
    labels = measure.label(segmentation)
    labels_info = measure.regionprops(labels)
    labels_info = [label for label in labels_info if label.area > pixel_cut_off]
    centroids = np.array([label.centroid for label in labels_info])
    area = np.array([label.area for label in labels_info])
    coords = np.array([label.coords for label in labels_info], dtype=object)
    return centroids, area, coords


def update_spacing(anchoring, width, height, grid_spacing):
    """
    Calculates spacing along width and height from slice anchoring.

    Args:
        anchoring (list): Anchoring transformation parameters.
        width (int): Image width.
        height (int): Image height.
        grid_spacing (int): Grid spacing in image units.

    Returns:
        tuple: (xspacing, yspacing)
    """
    if len(anchoring) != 9:
        print("Anchoring does not have 9 elements.")
    ow = np.sqrt(sum([anchoring[i + 3] ** 2 for i in range(3)]))
    oh = np.sqrt(sum([anchoring[i + 6] ** 2 for i in range(3)]))
    xspacing = int(width * grid_spacing / ow)
    yspacing = int(height * grid_spacing / oh)
    return xspacing, yspacing


def create_damage_mask(section, grid_spacing):
    """
    Creates a binary damage mask from grid information in the given section.

    Args:
        section (dict): Dictionary with slice and grid data.
        grid_spacing (int): Space between grid marks.

    Returns:
        ndarray: Binary mask with damaged areas marked as 0.
    """
    width = section["width"]
    height = section["height"]
    anchoring = section["anchoring"]
    grid_values = section["grid"]
    gridx = section["gridx"]
    gridy = section["gridy"]

    xspacing, yspacing = update_spacing(anchoring, width, height, grid_spacing)
    x_coords = np.arange(gridx, width, xspacing)
    y_coords = np.arange(gridy, height, yspacing)

    num_markers = len(grid_values)
    markers = [
        (x_coords[i % len(x_coords)], y_coords[i // len(x_coords)])
        for i in range(num_markers)
    ]

    binary_image = np.ones((len(y_coords), len(x_coords)), dtype=int)

    for i, (x, y) in enumerate(markers):
        if grid_values[i] == 4:
            binary_image[y // yspacing, x // xspacing] = 0

    return binary_image


def folder_to_atlas_space(
    folder,
    quint_alignment,
    atlas_labels,
    pixel_id=[0, 0, 0],
    non_linear=True,
    object_cutoff=0,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    apply_damage_mask=True,
):
    """
    Processes all segmentation files in a folder, mapping each one to atlas space.

    Args:
        folder (str): Path to segmentation files.
        quint_alignment (str): Path to alignment JSON.
        atlas_labels (DataFrame): DataFrame with atlas labels.
        pixel_id (list, optional): Pixel color to match.
        non_linear (bool, optional): Apply non-linear transform.
        object_cutoff (int, optional): Minimum object size.
        atlas_volume (ndarray, optional): Atlas volume data.
        hemi_map (ndarray, optional): Hemisphere mask data.
        use_flat (bool, optional): If True, load flat files.
        apply_damage_mask (bool, optional): If True, apply damage mask.

    Returns:
        tuple: Various arrays and lists containing transformed coordinates and labels.
    """
    quint_json = load_quint_json(quint_alignment)
    slices = quint_json['slices']
    if apply_damage_mask and "gridspacing" in quint_json:
        gridspacing = quint_json["gridspacing"]
    else:
        gridspacing = None
    if not apply_damage_mask:
        for slice in slices:
            if "grid" in slice:
                slice.pop("grid")
    segmentations = get_segmentations(folder)
    flat_files, flat_file_nrs = get_flat_files(folder, use_flat)
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
    ] * len(segmentations)
    points_list = [np.array([])] * len(segmentations)
    points_labels = [np.array([])] * len(segmentations)
    centroids_list = [np.array([])] * len(segmentations)
    centroids_labels = [np.array([])] * len(segmentations)
    per_point_undamaged_list = [np.array([])] * len(segmentations)
    per_centroid_undamaged_list = [np.array([])] * len(segmentations)
    points_hemi_labels = [np.array([])] * len(segmentations)
    centroids_hemi_labels = [np.array([])] * len(segmentations)
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
        centroids_labels,
        points_labels,
        region_areas_list,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
        points_hemi_labels,
        centroids_hemi_labels,
        object_cutoff,
        atlas_volume,
        hemi_map,
        use_flat,
        gridspacing,
    )
    start_and_join_threads(threads)
    (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        points_len,
        centroids_len,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    ) = process_results(
        points_list,
        centroids_list,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    )
    return (
        points,
        centroids,
        points_labels,
        centroids_labels,
        points_hemi_labels,
        centroids_hemi_labels,
        region_areas_list,
        points_len,
        centroids_len,
        segmentations,
        per_point_undamaged_list,
        per_centroid_undamaged_list,
    )


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
    centroids_labels,
    points_labels,
    region_areas_list,
    per_point_undamaged_list,
    per_centroid_undamaged_list,
    point_hemi_labels,
    centroid_hemi_labels,
    object_cutoff,
    atlas_volume,
    hemi_map,
    use_flat,
    gridspacing,
):
    """
    Creates threads to transform each segmentation into atlas space.

    Args:
        segmentations (list): Paths to segmentation files.
        slices (list): Slice metadata from alignment JSON.
        flat_files (list): Flat file paths for optional flat maps.
        flat_file_nrs (list): Numeric indices for flat files.
        atlas_labels (DataFrame): Atlas labels.
        pixel_id (list): Pixel color defined as [R, G, B].
        non_linear (bool): Enable non-linear transformation if True.
        points_list (list): Stores point coordinates per segmentation.
        centroids_list (list): Stores centroid coordinates per segmentation.
        centroids_labels (list): Stores labels for each centroid array.
        points_labels (list): Stores labels for each point array.
        region_areas_list (list): Stores region area data per segmentation.
        per_point_undamaged_list (list): Track undamaged points per segmentation.
        per_centroid_undamaged_list (list): Track undamaged centroids per segmentation.
        point_hemi_labels (list): Hemisphere labels for points.
        centroid_hemi_labels (list): Hemisphere labels for centroids.
        object_cutoff (int): Minimum object size threshold.
        atlas_volume (ndarray): 3D atlas volume (optional).
        hemi_map (ndarray): Hemisphere mask (optional).
        use_flat (bool): Use flat files if True.
        gridspacing (int): Spacing value from alignment data.

    Returns:
        list: A list of threads for parallel execution.
    """
    threads = []
    for segmentation_path, index in zip(segmentations, range(len(segmentations))):
        seg_nr = int(number_sections([segmentation_path])[0])
        current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
        if len(current_slice_index[0]) == 0:
            print("segmentation file does not exist in alignment json:")
            print(segmentation_path)
            continue
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
                points_labels,
                centroids_labels,
                region_areas_list,
                per_point_undamaged_list,
                per_centroid_undamaged_list,
                point_hemi_labels,
                centroid_hemi_labels,
                index,
                object_cutoff,
                atlas_volume,
                hemi_map,
                use_flat,
                gridspacing,
            ),
        )
        threads.append(x)
    return threads


def load_segmentation(segmentation_path: str):
    """
    Loads segmentation data, handling .dzip files if necessary.

    Args:
        segmentation_path (str): File path.

    Returns:
        ndarray: Image array of the segmentation.
    """
    if segmentation_path.endswith(".dzip"):
        return reconstruct_dzi(segmentation_path)
    else:
        return cv2.imread(segmentation_path)


def detect_pixel_id(segmentation: np.array):
    """
    Infers pixel color from the first non-background region.

    Args:
        segmentation (ndarray): Segmentation array.

    Returns:
        ndarray: Identified pixel color (RGB).
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
    hemi_mask,
    triangulation,
    damage_mask,
):
    """
    Builds the atlas map for a slice and calculates the region areas.

    Args:
        use_flat (bool): If True, uses flat files.
        atlas_labels (DataFrame): DataFrame containing atlas labels.
        flat_file_atlas (str): Path to the flat atlas file.
        seg_width (int): Segmentation image width.
        seg_height (int): Segmentation image height.
        slice_dict (dict): Dictionary with slice metadata (anchoring, etc.).
        atlas_volume (ndarray): 3D atlas volume.
        hemi_mask (ndarray): Hemisphere mask.
        triangulation (ndarray): Triangulation data for non-linear transforms.
        damage_mask (ndarray): Binary damage mask.

    Returns:
        tuple: (DataFrame of region areas, atlas map array).
    """
    atlas_map = load_image(
        flat_file_atlas,
        slice_dict["anchoring"],
        atlas_volume,
        triangulation,
        (seg_width, seg_height),
        atlas_labels,
    )
    region_areas = flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    return region_areas, atlas_map


def segmentation_to_atlas_space(
    slice_dict,
    segmentation_path,
    atlas_labels,
    flat_file_atlas=None,
    pixel_id="auto",
    non_linear=True,
    points_list=None,
    centroids_list=None,
    points_labels=None,
    centroids_labels=None,
    region_areas_list=None,
    per_point_undamaged_list=None,
    per_centroid_undamaged_list=None,
    points_hemi_labels=None,
    centroids_hemi_labels=None,
    index=None,
    object_cutoff=0,
    atlas_volume=None,
    hemi_map=None,
    use_flat=False,
    grid_spacing=None,
):
    """
    Transforms a single segmentation file into atlas space.

    Args:
        slice_dict (dict): Slice information from alignment JSON.
        segmentation_path (str): Path to the segmentation file.
        atlas_labels (DataFrame): Atlas labels.
        flat_file_atlas (str, optional): Path to flat atlas, if using flat files.
        pixel_id (str or list, optional): Pixel color or 'auto'.
        non_linear (bool, optional): Use non-linear transforms if True.
        points_list (list, optional): Storage for transformed point coordinates.
        centroids_list (list, optional): Storage for transformed centroid coordinates.
        points_labels (list, optional): Storage for assigned point labels.
        centroids_labels (list, optional): Storage for assigned centroid labels.
        region_areas_list (list, optional): Storage for region area data.
        per_point_undamaged_list (list, optional): Track undamaged points.
        per_centroid_undamaged_list (list, optional): Track undamaged centroids.
        points_hemi_labels (list, optional): Hemisphere labels for points.
        centroids_hemi_labels (list, optional): Hemisphere labels for centroids.
        index (int, optional): Index in the lists.
        object_cutoff (int, optional): Minimum object size.
        atlas_volume (ndarray, optional): 3D atlas volume.
        hemi_map (ndarray, optional): Hemisphere mask.
        use_flat (bool, optional): Indicates use of flat files.
        grid_spacing (int, optional): Spacing value for damage mask.

    Returns:
        None
    """
    segmentation = load_segmentation(segmentation_path)
    if pixel_id == "auto":
        pixel_id = detect_pixel_id(segmentation)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    triangulation = get_triangulation(slice_dict, reg_width, reg_height, non_linear)
    if "grid" in slice_dict:
        damage_mask = create_damage_mask(slice_dict, grid_spacing)
    else:
        damage_mask = None
    if hemi_map is not None:
        hemi_mask = generate_target_slice(slice_dict["anchoring"], hemi_map)
    else:
        hemi_mask = None
    region_areas, atlas_map = get_region_areas(
        use_flat,
        atlas_labels,
        flat_file_atlas,
        seg_width,
        seg_height,
        slice_dict,
        atlas_volume,
        hemi_mask,
        triangulation,
        damage_mask,
    )
    atlas_map = rescale_image(atlas_map, (reg_height, reg_width))
    y_scale, x_scale = transform_to_registration(
        seg_width, seg_height, reg_width, reg_height
    )
    centroids, points = None, None
    scaled_centroidsX, scaled_centroidsY, scaled_x, scaled_y = None, None, None, None
    centroids, scaled_centroidsX, scaled_centroidsY = get_centroids(
        segmentation, pixel_id, y_scale, x_scale, object_cutoff
    )
    scaled_y, scaled_x = get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale)
    per_point_labels = atlas_map[
        np.round(scaled_y).astype(int), np.round(scaled_x).astype(int)
    ]
    per_centroid_labels = atlas_map[
        np.round(scaled_centroidsY).astype(int), np.round(scaled_centroidsX).astype(int)
    ]
    if damage_mask is not None:
        damage_mask = cv2.resize(
            damage_mask.astype(np.uint8),
            (atlas_map.shape[::-1]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        per_point_undamaged = damage_mask[
            np.round(scaled_y).astype(int), np.round(scaled_x).astype(int)
        ]
        per_centroid_undamaged = damage_mask[
            np.round(scaled_centroidsY).astype(int),
            np.round(scaled_centroidsX).astype(int),
        ]
    else:
        per_point_undamaged = np.ones(scaled_x.shape, dtype=bool)
        per_centroid_undamaged = np.ones(scaled_centroidsX.shape, dtype=bool)
    if hemi_mask is not None:
        hemi_mask = cv2.resize(
            hemi_mask.astype(np.uint8),
            (atlas_map.shape[::-1]),
            interpolation=cv2.INTER_NEAREST,
        )

        per_point_hemi = hemi_mask[
            np.round(scaled_y).astype(int),
            np.round(scaled_x).astype(int),
        ]
        per_centroid_hemi = hemi_mask[
            np.round(scaled_centroidsY).astype(int),
            np.round(scaled_centroidsX).astype(int),
        ]
        per_point_hemi = per_point_hemi[per_point_undamaged]
        per_centroid_hemi = per_centroid_hemi[per_centroid_undamaged]
    else:
        per_point_hemi = [None] * len(scaled_x)
        per_centroid_hemi = [None] * len(scaled_centroidsX)

    per_point_labels = per_point_labels[per_point_undamaged]
    per_centroid_labels = per_centroid_labels[per_centroid_undamaged]

    new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(
        non_linear,
        slice_dict,
        scaled_x[per_point_undamaged],
        scaled_y[per_point_undamaged],
        scaled_centroidsX[per_centroid_undamaged],
        scaled_centroidsY[per_centroid_undamaged],
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
    centroids_labels[index] = np.array(
        per_centroid_labels if centroids is not None else []
    )
    per_centroid_undamaged_list[index] = np.array(
        per_centroid_undamaged if centroids is not None else []
    )
    points_labels[index] = np.array(per_point_labels if points is not None else [])
    per_point_undamaged_list[index] = np.array(
        per_point_undamaged if points is not None else []
    )
    points_hemi_labels[index] = np.array(per_point_hemi if points is not None else [])
    centroids_hemi_labels[index] = np.array(
        per_centroid_hemi if points is not None else []
    )


def get_triangulation(slice_dict, reg_width, reg_height, non_linear):
    """
    Generates triangulation data if non-linear markers exist.

    Args:
        slice_dict (dict): Slice metadata from alignment JSON.
        reg_width (int): Registration width.
        reg_height (int): Registration height.
        non_linear (bool): Whether to use non-linear transform.

    Returns:
        list or None: Triangulation info or None if not applicable.
    """
    if non_linear and "markers" in slice_dict:
        return triangulate(reg_width, reg_height, slice_dict["markers"])
    return None


def get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff=0):
    """
    Finds object centroids for a given pixel color and applies scaling.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel color to match.
        y_scale (float): Vertical scaling factor.
        x_scale (float): Horizontal scaling factor.
        object_cutoff (int, optional): Minimum object size.

    Returns:
        tuple: (centroids, scaled_centroidsX, scaled_centroidsY)
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
    Retrieves pixel coordinates for a specified color and scales them.

    Args:
        segmentation (ndarray): Segmentation array.
        pixel_id (int): Pixel color to match.
        y_scale (float): Vertical scaling factor.
        x_scale (float): Horizontal scaling factor.

    Returns:
        tuple: (scaled_y, scaled_x)
    """
    id_pixels = find_matching_pixels(segmentation, pixel_id)
    if len(id_pixels[0]) == 0:
        return None, None
    scaled_y, scaled_x = scale_positions(id_pixels[0], id_pixels[1], y_scale, x_scale)
    return scaled_y, scaled_x
