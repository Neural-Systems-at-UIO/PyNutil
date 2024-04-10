import os
import sys
import numpy as np
import tifffile as tiff
path_to_file = r"/home/harryc/Downloads/mouse-s23/ext-d000009_PVMouse_81265_Samp1__s023.tif"
image_outdir = path_to_file.split(".")[0].split("/")[-1]
if not os.path.exists(image_outdir):
    os.makedirs(image_outdir)
# Read the image
img = tiff.imread(path_to_file)
# Get the shape of the image
shape = img.shape
#Get the chunk size
chunk_size = 512
#invert the image if neccessary
img = np.invert(img)
#crop the image and save each chunk
for i in range(0, shape[0], chunk_size):
    for j in range(0, shape[1], chunk_size):
        chunk = img[i:i+chunk_size, j:j+chunk_size]
        tiff.imsave("{}/chunk_{}_{}.tif".format(image_outdir,i, j), chunk)
