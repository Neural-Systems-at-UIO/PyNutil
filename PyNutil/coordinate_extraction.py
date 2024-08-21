import numpy as np
import pandas as pd
from .read_and_write import load_visualign_json
from .counting_and_load import flat_to_dataframe
from .visualign_deformations import triangulate, transform_vec
from glob import glob
import cv2
from skimage import measure
import threading
import re
from .reconstruct_dzi import reconstruct_dzi


def number_sections(filenames, legacy=False):
    """
    returns the section numbers of filenames

    :param filenames: list of filenames
    :type filenames: list[str]
    :return: list of section numbers
    :rtype: list[int]
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
            ###this gets the three numbers closest to the end
            section_numbers.append(match[-3:])
    if len(section_numbers) == 0:
        raise ValueError("No section numbers found in filenames")
    return section_numbers


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
    atlas_volume=None,
    use_flat=False,
):
    """Apply Segmentation to atlas space to all segmentations in a folder."""
    """Return pixel_points, centroids, points_len, centroids_len, segmentation_filenames, """
    # This should be loaded above and passed as an argument
    slices = load_visualign_json(quint_alignment)

    segmentation_file_types = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dzip"]
    segmentations = [
        file
        for file in glob(folder + "/segmentations/*")
        if any([file.endswith(type) for type in segmentation_file_types])
    ]
    if len(segmentations) == 0:
        raise ValueError(
            f"No segmentations found in folder {folder}. Make sure the folder contains a segmentations folder with segmentations."
        )
    print(f"Found {len(segmentations)} segmentations in folder {folder}")
    if use_flat == True:
        flat_files = [
            file
            for file in glob(folder + "/flat_files/*")
            if any([file.endswith(".flat"), file.endswith(".seg")])
        ]
        print(f"Found {len(flat_files)} flat files in folder {folder}")
        flat_file_nrs = [int(number_sections([ff])[0]) for ff in flat_files]

    # Order segmentations and section_numbers
    # segmentations = [x for _,x in sorted(zip(section_numbers,segmentations))]
    # section_numbers.sort()
    points_list = [np.array([])] * len(segmentations)
    centroids_list = [np.array([])] * len(segmentations)
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
    threads = []
    for segmentation_path, index in zip(segmentations, range(len(segmentations))):
        seg_nr = int(number_sections([segmentation_path])[0])
        current_slice_index = np.where([s["nr"] == seg_nr for s in slices])
        current_slice = slices[current_slice_index[0][0]]
        if current_slice["anchoring"] == []:
            continue
        if use_flat == True:
            current_flat_file_index = np.where([f == seg_nr for f in flat_file_nrs])
            current_flat = flat_files[current_flat_file_index[0][0]]
        else:
            current_flat = None

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
                atlas_volume,
                use_flat,
            ),
        )
        threads.append(x)
        ## This converts the segmentation to a point cloud
    # Start threads
    [t.start() for t in threads]
    # Wait for threads to finish
    [t.join() for t in threads]
    # Flatten points_list

    points_len = [
        len(points) if None not in points else 0 for points in points_list
        ]
    centroids_len = [
             len(centroids) if None not in centroids else 0 for centroids in centroids_list
         ]
    points_list = [points for points in points_list if None not in points]
    centroids_list = [centroids for centroids in centroids_list if None not in centroids]
    if len(points_list) == 0:
        points = np.array([])
    else:
        points = np.concatenate(points_list)
    if len(centroids_list) == 0:
        centroids = np.array([])
    else:
        centroids = np.concatenate(centroids_list)


    return (
        np.array(points),
        np.array(centroids),
        region_areas_list,
        points_len,
        centroids_len,
        segmentations,
    )

def load_segmentation(segmentation_path: str):
    """Load a segmentation from a file."""
    print(f"working on {segmentation_path}")
    if segmentation_path.endswith(".dzip"):
        print("Reconstructing dzi")
        return reconstruct_dzi(segmentation_path)
    else:
        return cv2.imread(segmentation_path)

def detect_pixel_id(segmentation: np.array):
    """Remove the background from the segmentation and return the pixel id."""
    segmentation_no_background = segmentation[~np.all(segmentation == 0, axis=2)]
    pixel_id = segmentation_no_background[0]
    print("detected pixel_id: ", pixel_id)
    return pixel_id

def get_region_areas(use_flat, atlas_labels, flat_file_atlas, seg_width, seg_height, slice_dict, atlas_volume, triangulation):
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
            triangulation
        )
    return region_areas

def get_transformed_coordinates(non_linear, slice_dict, method, scaled_x, scaled_y, centroids, scaled_centroidsX, scaled_centroidsY, triangulation):
    new_x, new_y, centroids_new_x, centroids_new_y = None, None, None, None
    if non_linear and "markers" in slice_dict:
        if method in ["per_pixel", "all"] and scaled_x is not None:
            new_x, new_y = transform_vec(triangulation, scaled_x, scaled_y)
        if method in ["per_object", "all"] and centroids is not None:
            centroids_new_x, centroids_new_y = transform_vec(triangulation, scaled_centroidsX, scaled_centroidsY)
    else:
        if method in ["per_pixel", "all"]:
            new_x, new_y = scaled_x, scaled_y
        if method in ["per_object", "all"]:
            centroids_new_x, centroids_new_y = scaled_centroidsX, scaled_centroidsY
    return new_x, new_y, centroids_new_x, centroids_new_y

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
    method="per_pixel",
    object_cutoff=0,
    atlas_volume=None,
    use_flat=False,
):
    segmentation = load_segmentation(segmentation_path)
    if pixel_id == "auto":
        pixel_id = detect_pixel_id(segmentation)
    seg_height, seg_width = segmentation.shape[:2]
    reg_height, reg_width = slice_dict["height"], slice_dict["width"]
    if non_linear and "markers" in slice_dict:
        triangulation = triangulate(reg_width, reg_height, slice_dict["markers"])
    else:
        triangulation = None
    region_areas = get_region_areas(use_flat, atlas_labels, flat_file_atlas, seg_width, seg_height, slice_dict, atlas_volume, triangulation)
    y_scale, x_scale = transform_to_registration(seg_height, seg_width, reg_height, reg_width)
    centroids, points = None, None
    scaled_centroidsX, scaled_centroidsY, scaled_x, scaled_y = None, None, None, None 
    if method in ["per_object", "all"]:
        centroids, scaled_centroidsX, scaled_centroidsY = get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff)
    if method in ["per_pixel", "all"]:
        scaled_y, scaled_x = get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale)

    new_x, new_y, centroids_new_x, centroids_new_y = get_transformed_coordinates(non_linear, slice_dict, method, scaled_x, scaled_y, centroids, scaled_centroidsX, scaled_centroidsY, triangulation)
    if method in ["per_pixel", "all"] and new_x is not None:
        points = transform_to_atlas_space(slice_dict["anchoring"], new_y, new_x, reg_height, reg_width)
    if method in ["per_object", "all"] and centroids_new_x is not None:
        centroids = transform_to_atlas_space(slice_dict["anchoring"], centroids_new_y, centroids_new_x, reg_height, reg_width)
    points_list[index] = np.array(points if points is not None else [])
    centroids_list[index] = np.array(centroids if centroids is not None else [])
    region_areas_list[index] = region_areas


def get_centroids(segmentation, pixel_id, y_scale, x_scale, object_cutoff=0):
    binary_seg = segmentation == pixel_id
    binary_seg = np.all(binary_seg, axis=2)
    centroids, area, coords = get_centroids_and_area(
        binary_seg, pixel_cut_off=object_cutoff
    )

    print(f"using pixel id {pixel_id}")
    print(f"Found {len(centroids)} objects in the segmentation")
    if len(centroids) == 0:
        return None, None, None
    centroidsX = centroids[:, 1]
    centroidsY = centroids[:, 0]
    scaled_centroidsY, scaled_centroidsX = scale_positions(
        centroidsY, centroidsX, y_scale, x_scale
    )
    return centroids, scaled_centroidsX, scaled_centroidsY


def get_scaled_pixels(segmentation, pixel_id, y_scale, x_scale):
    id_pixels = find_matching_pixels(segmentation, pixel_id)
    if len(id_pixels[0]) == 0:
        return None, None
    # Scale the seg coordinates to reg/seg
    scaled_y, scaled_x = scale_positions(id_pixels[0], id_pixels[1], y_scale, x_scale)
    return scaled_y, scaled_x
