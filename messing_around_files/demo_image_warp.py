import json
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import nrrd
atlas_path = r"/home/harryc/Github/PyNutilWeb/server/PyNutil/PyNutil/metadata/annotation_volumes/annotation_25_reoriented_2017.nrrd"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyNutil.generate_target_slice import generate_target_slice
from PyNutil.visualign_deformations import triangulate, transform_vec

def make_slice_ordinal(data):
    for i, slice in enumerate(np.unique(data)):
        data[data==slice] = i
    return data
data_path = r"/home/harryc/Github/PyNutilWeb/server/PyNutil/test_data/PyNutil_testdataset_Nonlin_SY_fixed_bigcaudoputamen.json"
with open(data_path, "r") as f:
    data = json.load(f)

volume, _ = nrrd.read(atlas_path)
demo = data["slices"][0]
demo_alignment = demo["anchoring"]
demo_markers = demo["markers"]
h = demo["height"]
w = demo["width"]


image = generate_target_slice(demo_alignment, volume)
image = make_slice_ordinal(image)
plt.imshow(image)
plt.show()

triangulation = triangulate(w, h, demo_markers)





def warp_image(image, triangulation, h,w):
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

new_image = warp_image(image, triangulation, h, w)
plt.imshow(new_image)
plt.show()