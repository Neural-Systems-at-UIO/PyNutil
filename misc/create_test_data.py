"""Helper script to generate simple PNG test images."""

import os

import cv2
import numpy as np


def generate_test_image(width, height, file_path):
    """Create a plain white RGB image and save as PNG."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    # cv2 expects BGR
    ok = cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {file_path}")


"""This is used to generate the test data"""


def generate_image_with_squares(
    width, height, square_diameter, square_locations, num_images
):
    """Create a series of white images with filled black squares at given locations."""
    for i in range(1, num_images + 1):
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        # Draw black squares (filled)
        for location in square_locations:
            x, y = location
            x2 = x + (square_diameter - 1)
            y2 = y + (square_diameter - 1)
            cv2.rectangle(image, (x, y), (x2, y2), color=(0, 0, 0), thickness=-1)

        file_name = f"../test_data/PyTest/test_s00{i}.png"
        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
        ok = cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write image: {file_name}")


if __name__ == "__main__":
    # Example usage
    width = 1000  # Specify the width of the image in pixels
    height = 1000  # Specify the height of the image in pixels
    file_path = "testimage.png"  # Specify the file path for saving the image
    generate_test_image(width, height, file_path)

    width = 1500  # Specify the width of the image in pixels
    height = 1000  # Specify the height of the image in pixels
    square_diameter = 10  # Specify the size of the black squares in pixels
    square_locations = [
        (500, 500),
        (500, 600),
        (500, 700),
        (1000, 500),
        (1000, 600),
        (1000, 700),
    ]
    num_images = 5
    generate_image_with_squares(
        width, height, square_diameter, square_locations, num_images
    )
