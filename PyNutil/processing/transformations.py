import numpy as np
from .visualign_deformations import transform_vec


def transform_to_registration(seg_height, seg_width, reg_height, reg_width):
    """
    Returns the scaling factors to transform the segmentation to the registration space.

    Args:
        seg_height (int): Segmentation height.
        seg_width (int): Segmentation width.
        reg_height (int): Registration height.
        reg_width (int): Registration width.

    Returns:
        tuple: Y and X scaling factors.
    """
    y_scale = reg_height / seg_height
    x_scale = reg_width / seg_width
    return y_scale, x_scale


def transform_to_atlas_space(anchoring, y, x, reg_height, reg_width):
    """
    Transforms to atlas space using the QuickNII anchoring vector.

    Args:
        anchoring (list): Anchoring vector.
        y (ndarray): Y coordinates.
        x (ndarray): X coordinates.
        reg_height (int): Registration height.
        reg_width (int): Registration width.

    Returns:
        ndarray: Transformed coordinates.
    """
    o = anchoring[0:3]
    u = anchoring[3:6]
    u = np.array([u[0], u[1], u[2]])
    v = anchoring[6:9]
    v = np.array([v[0], v[1], v[2]])
    y_scale = y / reg_height
    x_scale = x / reg_width
    xyz_v = np.array([y_scale * v[0], y_scale * v[1], y_scale * v[2]])
    xyz_u = np.array([x_scale * u[0], x_scale * u[1], x_scale * u[2]])
    o = np.reshape(o, (3, 1))
    return (o + xyz_u + xyz_v).T

def image_to_atlas_space(image, anchoring):
    """
    Transforms to atlas space an image using the QuickNII anchoring vector.


    Args:
        image (ndarray): An Image which you would like to apply the anchoring vector to
        anchoring (list): Anchoring vector.

    Returns:
        ndarray: Transformed coordinates for every pixel in the image.
    """
    width = image.shape[1]
    height = image.shape[0]
    x = np.arange(width)
    y = np.arange(height)
    x_coords, y_coords = np.meshgrid(x, y)
    coordinates = transform_to_atlas_space(anchoring, y_coords.flatten(), x_coords.flatten(), height, width)
    return coordinates

def get_transformed_coordinates(
    non_linear,
    slice_dict,
    scaled_x,
    scaled_y,
    scaled_centroidsX,
    scaled_centroidsY,
    triangulation,
):
    """
    Compute transformed coordinates for both points and centroids.

    Args:
        non_linear (bool): Flag to indicate if non-linear transformation is applied.
        slice_dict (dict): Slice metadata including markers.
        scaled_x (ndarray): Scaled x coordinates for points.
        scaled_y (ndarray): Scaled y coordinates for points.
        scaled_centroidsX (ndarray): Scaled x coordinates for centroids.
        scaled_centroidsY (ndarray): Scaled y coordinates for centroids.
        triangulation (list): Triangulation structure to be used if non_linear is True.

    Returns:
    Returns:
        tuple: Transformed coordinates.
    """
    new_x, new_y, centroids_new_x, centroids_new_y = None, None, None, None
    if non_linear and "markers" in slice_dict:
        if scaled_x is not None:
            new_x, new_y = transform_vec(triangulation, scaled_x, scaled_y)
        if scaled_centroidsX is not None:
            centroids_new_x, centroids_new_y = transform_vec(
                triangulation, scaled_centroidsX, scaled_centroidsY
            )
    else:
        new_x, new_y = scaled_x, scaled_y
        centroids_new_x, centroids_new_y = scaled_centroidsX, scaled_centroidsY
    return new_x, new_y, centroids_new_x, centroids_new_y


def transform_points_to_atlas_space(
    slice_dict, new_x, new_y, centroids_new_x, centroids_new_y, reg_height, reg_width
):
    """
    Transforms points and centroids to atlas space.

    Args:
        slice_dict (dict): Dictionary with slice information.
        new_x (ndarray): Transformed X coordinates.
        new_y (ndarray): Transformed Y coordinates.
        centroids_new_x (ndarray): Transformed X coordinates of centroids.
        centroids_new_y (ndarray): Transformed Y coordinates of centroids.
        reg_height (int): Registration height.
        reg_width (int): Registration width.

    Returns:
        tuple: Transformed points and centroids.
    """
    points, centroids = None, None
    if new_x is not None:
        points = transform_to_atlas_space(
            slice_dict["anchoring"], new_y, new_x, reg_height, reg_width
        )
    if centroids_new_x is not None:
        centroids = transform_to_atlas_space(
            slice_dict["anchoring"],
            centroids_new_y,
            centroids_new_x,
            reg_height,
            reg_width,
        )
    return points, centroids
