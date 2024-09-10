import numpy as np
import pandas as pd
import struct
import cv2
from .generate_target_slice import generate_target_slice
from .visualign_deformations import transform_vec

# related to counting and load
def label_points(points, label_volume, scale_factor=1):
    """This function takes a list of points and assigns them to a region based on the region_volume.
    These regions will just be the values in the region_volume at the points.
    It returns a dictionary with the region as the key and the points as the value."""
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
    """Function for counting no. of pixels per region and writing to CSV based on
    a dictionary with the region as the key and the points as the value."""
    if labels_dict_points is not None and labeled_dict_centroids is not None:
        counted_labels_points, label_counts_points = np.unique(
            labels_dict_points, return_counts=True
        )
        counted_labels_centroids, label_counts_centroids = np.unique(
            labeled_dict_centroids, return_counts=True
        )
        # Which regions have pixels, and how many pixels are there per region
        counts_per_label = list(
            zip(counted_labels_points, label_counts_points, label_counts_centroids)
        )
        # Create a list of unique regions and pixel counts per region
        df_counts_per_label = pd.DataFrame(
            counts_per_label, columns=["idx", "pixel_count", "object_count"]
        )
    elif labels_dict_points is None and labeled_dict_centroids is not None:
        counted_labels_centroids, label_counts_centroids = np.unique(
            labeled_dict_centroids, return_counts=True
        )
        # Which regions have pixels, and how many pixels are there per region
        counts_per_label = list(zip(counted_labels_centroids, label_counts_centroids))
        # Create a list of unique regions and pixel counts per region
        df_counts_per_label = pd.DataFrame(
            counts_per_label, columns=["idx", "object_count"]
        )
    elif labels_dict_points is not None and labeled_dict_centroids is None:
        counted_labels_points, label_counts_points = np.unique(
            labels_dict_points, return_counts=True
        )
        # Which regions have pixels, and how many pixels are there per region
        counts_per_label = list(zip(counted_labels_points, label_counts_points))
        # Create a list of unique regions and pixel counts per region
        df_counts_per_label = pd.DataFrame(
            counts_per_label, columns=["idx", "pixel_count"]
        )
    # Create a pandas df with regions and pixel counts

    # df_label_colours = pd.read_csv(label_colours, sep=",")
    # Find colours corresponding to each region ID and add to the pandas dataframe

    # Look up name, r, g, b in df_allen_colours in df_counts_per_label based on "idx"
    # Sharon, remove this here
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
    w, h = rescaleXY
    return cv2.resize(image, (h, w), interpolation=cv2.INTER_NEAREST)

def assign_labels_to_image(image, labelfile):
    w, h = image.shape
    allen_id_image = np.zeros((h, w))  # create an empty image array
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))

    values = image[coordsy, coordsx]
    lbidx = labelfile["idx"].values

    allen_id_image = lbidx[values.astype(int)]
    return allen_id_image


def count_pixels_per_label(image, scale_factor=False):
    unique_ids, counts = np.unique(image, return_counts=True)
    if scale_factor:
        counts = counts * scale_factor
    area_per_label = list(zip(unique_ids, counts))
    df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "region_area"])
    return df_area_per_label


def warp_image(image, triangulation, rescaleXY):
    if rescaleXY is not None:
        w,h = rescaleXY
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
    newX[newX >= reg_w] = reg_w - 1
    newY[newY >= reg_h] = reg_h - 1
    newX[newX < 0] = 0
    newY[newY < 0] = 0
    new_image = image[newY, newX]
    return new_image

def flat_to_dataframe(
    labelfile, file=None, rescaleXY=None, image_vector=None, volume=None, triangulation=None
):
    if (image_vector is not None) and (volume is not None):
        image = generate_target_slice(image_vector, volume)
        image = np.float64(image)
        if triangulation is not None:
            image = warp_image(image, triangulation, rescaleXY)
    elif file.endswith(".flat"):
        image = read_flat_file(file)
    elif file.endswith(".seg"):
        image = read_seg_file(file)
    print("datatype", image.dtype)
    print("image shape open", image.shape)

    if rescaleXY:
        image_shapeY, image_shapeX = image.shape[0], image.shape[1]
        image_pixels = image_shapeY * image_shapeX
        seg_pixels = rescaleXY[0] * rescaleXY[1]
        scale_factor = seg_pixels / image_pixels
    else:
        scale_factor = False
    if (image_vector is None) or (volume is None):
        allen_id_image = assign_labels_to_image(image, labelfile)
    else:
        allen_id_image = image
    df_area_per_label = count_pixels_per_label(allen_id_image, scale_factor)
    return df_area_per_label
