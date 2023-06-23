import numpy as np
import pandas as pd
from DeepSlice.coord_post_processing.spacing_and_indexing import number_sections
import json
from .read_and_write import load_visualign_json
from .counting_and_load import label_points, flat_to_dataframe
from .visualign_deformations import triangulate, transform_vec
from glob import glob
from tqdm import tqdm
import cv2
from skimage import measure
import threading


# related to coordinate_extraction
def get_centroids_and_area(segmentation, pixel_cut_off=0):
    """This function returns the center coordinate of each object in the segmentation.
    You can set a pixel_cut_off to remove objects that are smaller than that number of pixels.
    """
    labels = measure.label(segmentation)
    # This finds all the objects in the image
    labels_info = measure.regionprops(labels)

    # Remove objects that are less than pixel_cut_off
    labels_info = [label for label in labels_info if label.area > pixel_cut_off]
    # Get the centre points of the objects
    centroids = np.array([label.centroid for label in labels_info])
    # Get the area of the objects
    area = np.array([label.area for label in labels_info])
    # Get the coordinates for all the pixels in each object
    coords = np.array([label.coords for label in labels_info], dtype=object)
    return centroids, area, coords


# related to coordinate extraction
def transform_to_registration(seg_height, seg_width, reg_height, reg_width):
    """This function returns the scaling factors to transform the segmentation to the registration space."""
    y_scale = reg_height / seg_height
    x_scale = reg_width / seg_width
    return y_scale, x_scale


# related to coordinate extraction
def find_matching_pixels(segmentation, id):
    """This function returns the Y and X coordinates of all the pixels in the segmentation that match the id provided."""
    mask = segmentation == id
    mask = np.all(mask, axis=2)
    id_positions = np.where(mask)
    id_y, id_x = id_positions[0], id_positions[1]
    return id_y, id_x


# related to coordinate extraction
def scale_positions(id_y, id_x, y_scale, x_scale):
    """This function scales the Y and X coordinates to the registration space.
    (The y_scale and x_scale are the output of transform_to_registration.)
    """
    id_y = id_y * y_scale
    id_x = id_x * x_scale
    return id_y, id_x


# related to coordinate extraction
def transform_to_atlas_space(anchoring, y, x, reg_height, reg_width):
    """Transform to atlas space using the QuickNII anchoring vector."""
    o = anchoring[0:3]
    u = anchoring[3:6]
    # Swap order of U
    u = np.array([u[0], u[1], u[2]])
    v = anchoring[6:9]
    # Swap order of V
    v = np.array([v[0], v[1], v[2]])
    # Scale X and Y to between 0 and 1 using the registration width and height
    y_scale = y / reg_height
    x_scale = x / reg_width
    xyz_v = np.array([y_scale * v[0], y_scale * v[1], y_scale * v[2]])
    xyz_u = np.array([x_scale * u[0], x_scale * u[1], x_scale * u[2]])
    o = np.reshape(o, (3, 1))
    return (o + xyz_u + xyz_v).T


# points.append would make list of lists, keeping sections separate.


# related to coordinate extraction
# This function returns an array of points
def folder_to_atlas_space(
    folder,
    quint_alignment,
    atlas_labels,
    pixel_id=[0, 0, 0],
    non_linear=True,
    method="all",
    object_cutoff=0,
):
    """Apply Segmentation to atlas space to all segmentations in a folder."""
    """Return pixel_points, centroids, points_len, centroids_len, segmentation_filenames, """
    # This should be loaded above and passed as an argument
    slices = load_visualign_json(quint_alignment)

    segmentation_file_types = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    segmentations = [
        file
        for file in glob(folder + "/*")
        if any([file.endswith(type) for type in segmentation_file_types])
    ]
    flat_files = [
        file
        for file in glob(folder + "/flat_files/*")
        if any([file.endswith('.flat')])
    ]
    # Order segmentations and section_numbers
    # segmentations = [x for _,x in sorted(zip(section_numbers,segmentations))]
    # section_numbers.sort()
    points_list = [None] * len(segmentations)
    centroids_list = [None] * len(segmentations)
    region_areas_list = [None] * len(segmentations)
    threads = []
    flat_file_nrs = [int(number_sections([ff])[0]) for ff in flat_files]
    for segmentation_path, index in zip(segmentations, range(len(segmentations))):
        seg_nr = int(number_sections([segmentation_path])[0])
        current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
        current_slice = slices[current_slice_index[0][0]]
        current_flat_file_index = np.where([f == seg_nr for f in flat_file_nrs])
        current_flat = flat_files[current_flat_file_index[0][0]]
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
                method,
                object_cutoff,
            ),
        )
        threads.append(x)
        ## This converts the segmentation to a point cloud
    # Start threads
    [t.start() for t in threads]
    # Wait for threads to finish
    [t.join() for t in threads]
    # Flatten points_list

    points_len = [len(points) for points in points_list]
    centroids_len = [len(centroids) for centroids in centroids_list]
    points = np.concatenate(points_list)
    centroids = np.concatenate(centroids_list)

    return (
        np.array(points),
        np.array(centroids),
        region_areas_list,
        points_len,
        centroids_len,
        segmentations,
    )


def segmentation_to_atlas_space(
    slice,
    segmentation_path,
    atlas_labels, 
    flat_file_atlas,
    pixel_id="auto",
    non_linear=True,
    points_list=None,
    centroids_list=None,
    region_areas_list=None, 
    index=None,
    method="per_pixel",
    object_cutoff=0,
    
):
    """Combines many functions to convert a segmentation to atlas space. It takes care
    of deformations."""
    segmentation = cv2.imread(segmentation_path)
    if pixel_id == "auto":
        # Remove the background from the segmentation
        segmentation_no_background = segmentation[~np.all(segmentation == 255, axis=2)]
        pixel_id = np.vstack(
            {tuple(r) for r in segmentation_no_background.reshape(-1, 3)}
        )  # Remove background
        # Currently only works for a single label
        pixel_id = pixel_id[0]

    # Transform pixels to registration space (the registered image and segmentation have different dimensions)
    seg_height = segmentation.shape[0]
    seg_width = segmentation.shape[1]
    reg_height = slice["height"]
    reg_width = slice["width"]
    region_areas = flat_to_dataframe(flat_file_atlas, atlas_labels, (seg_width,seg_height))
    # This calculates reg/seg
    y_scale, x_scale = transform_to_registration(
        seg_height, seg_width, reg_height, reg_width
    )
    centroids, points = None, None
    if method in ["per_object", "all"]:
        centroids, scaled_centroidsX, scaled_centroidsY = get_centroids(
            segmentation, pixel_id, y_scale, x_scale, object_cutoff
        )
    if method in ["per_pixel", "all"]:
        scaled_y, scaled_x = get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale)

    if non_linear:
        if "markers" in slice:
            # This creates a triangulation using the reg width
            triangulation = triangulate(reg_width, reg_height, slice["markers"])
            if method in ["per_pixel", "all"]:
                new_x, new_y = transform_vec(triangulation, scaled_x, scaled_y)
            if method in ["per_object", "all"]:
                centroids_new_x, centroids_new_y = transform_vec(
                    triangulation, scaled_centroidsX, scaled_centroidsY
                )
        else:
            print(
                f"No markers found for {slice['filename']}, result for section will be linear."
            )
            if method in ["per_pixel", "all"]:
                new_x, new_y = scaled_x, scaled_y
            if method in ["per_object", "all"]:
                centroids_new_x, centroids_new_y = scaled_centroidsX, scaled_centroidsY
    else:
        if method in ["per_pixel", "all"]:
            new_x, new_y = scaled_x, scaled_y
        if method in ["per_object", "all"]:
            centroids_new_x, centroids_new_y = scaled_centroidsX, scaled_centroidsY
    # Scale U by Uxyz/RegWidth and V by Vxyz/RegHeight
    if method in ["per_pixel", "all"]:
        points = transform_to_atlas_space(
            slice["anchoring"], new_y, new_x, reg_height, reg_width
        )
    if method in ["per_object", "all"]:
        centroids = transform_to_atlas_space(
            slice["anchoring"], centroids_new_y, centroids_new_x, reg_height, reg_width
        )
    print(
        f"Finished and points len is: {len(points)} and centroids len is: {len(centroids)}"
    )
    points_list[index] = np.array(points)
    centroids_list[index] = np.array(centroids)
    region_areas_list[index] = region_areas


def get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff=0):
    binary_seg = segmentation == pixel_id
    binary_seg = np.all(binary_seg, axis=2)
    centroids, area, coords = get_centroids_and_area(
        binary_seg, pixel_cut_off=object_cutoff
    )
    centroidsX = centroids[:, 1]
    centroidsY = centroids[:, 0]
    scaled_centroidsY, scaled_centroidsX = scale_positions(
        centroidsY, centroidsX, y_scale, x_scale
    )
    return centroids, scaled_centroidsX, scaled_centroidsY


def get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale):
    id_pixels = find_matching_pixels(segmentation, pixel_id)
    # Scale the seg coordinates to reg/seg
    scaled_y, scaled_x = scale_positions(id_pixels[0], id_pixels[1], y_scale, x_scale)
    return scaled_y, scaled_x
