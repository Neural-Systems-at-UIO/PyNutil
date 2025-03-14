import numpy as np
import matplotlib.pyplot as plt
from PyNutil.io.read_and_write import load_visualign_json
import cv2

path_to_json = r"/home/harryc/github/PyNutil/tests/test_data/nonlinear_allen_mouse/damage_markers.json"
dmg_json = load_visualign_json(path_to_json)
section = dmg_json[2]

width = section["width"]
height = section["height"]
gridx = section["gridx"]
gridy = section["gridy"]
anchoring = section["anchoring"]
grid_spacing = 20
grid_values = section["grid"]  # List of length 368 with mostly 0s and some 4s

def update_spacing(anchoring, width, height, grid_spacing):
    if len(anchoring) != 9:
        print("Anchoring does not have 9 elements.")
    ow = np.sqrt(sum([anchoring[i+3] ** 2 for i in range(3)]))
    oh = np.sqrt(sum([anchoring[i+6] ** 2 for i in range(3)]))
    xspacing = int(width * grid_spacing / ow)
    yspacing = int(height * grid_spacing / oh)
    return xspacing, yspacing

def create_damage_mask(section, grid_spacing):
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
    markers = [(x_coords[i % len(x_coords)], y_coords[i // len(x_coords)]) for i in range(num_markers)]

    binary_image = np.ones((len(y_coords), len(x_coords)), dtype=int)

    for i, (x, y) in enumerate(markers):
        if grid_values[i] == 4:
            binary_image[y // yspacing, x // xspacing] = 0

    return binary_image

binary_image = create_damage_mask(section, grid_spacing)
cv2.imwrite("binary.png", binary_image * 255)
plt.figure(figsize=(6, 6))
plt.imshow(binary_image, cmap='gray', origin='upper')
plt.title("Binary Image of Markers")
plt.show()