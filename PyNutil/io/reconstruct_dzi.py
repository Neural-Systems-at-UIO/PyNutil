import cv2
import numpy as np
import os
import zipfile
import xmltodict


def reconstruct_dzi(zip_file_path):
    """
    Reconstructs a Deep Zoom Image (DZI) from a zip file containing the tiles.
    Parameters
    ----------
    zip_file_path : str
        Path to the zip file containing the tiles.
    apply_damage_mask : bool
        Whether to apply the damage mask.

    Returns
    -------
    ndarray
       The reconstructed DZI.
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        # Get the highest level of the pyramid
        highest_level = str(
            np.max(
                [
                    int(os.path.split(os.path.split(i)[0])[1])
                    for i in zip_file.namelist()
                    if i.endswith(".png")
                ]
            )
        )

        # Get the filenames of the highest level tiles
        highest_level_files = [
            i
            for i in zip_file.namelist()
            if i.endswith(".png") and i.split("/")[-2] == highest_level
        ]

        # Read the DZI file
        dzi_file = zip_file.open(
            [i for i in zip_file.namelist() if i.endswith(".dzi")][0]
        )
        xml = dzi_file.read()
        json_data = xmltodict.parse(xml)
        tileSize = json_data["Image"]["@TileSize"]
        width, height = int(json_data["Image"]["Size"]["@Width"]), int(
            json_data["Image"]["Size"]["@Height"]
        )

        # Create an empty image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill in the image with the highest level tiles
        for file in highest_level_files:
            with zip_file.open(file) as f:
                contents = f.read()
            # Decode the binary PNG data
            image_ = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            x, y = map(int, os.path.splitext(os.path.split(file)[1])[0].split("_"))
            x, y = x * int(tileSize), y * int(tileSize)
            image[y : y + image_.shape[0], x : x + image_.shape[1], :] = image_

        # Save the image
    return image
