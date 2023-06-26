from PyNutil import generate_target_slice
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PyNutil.visualign_deformations import triangulate, transform_vec

volume_path = r"C:\Users\sharoncy\Documents\Github\PyNutil\PyNutil\metadata\annotation_volumes\annotation_25_reoriented_2017.nrrd"
volume, header = nrrd.read(volume_path)

alignment =[-5.145275115966797, 361.8014440433213, 331.1490739071843, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0]
alignment = np.array(alignment) 
section = generate_target_slice.generate_target_slice(alignment, volume)
unique_colours = np.unique(section)
color_map = {i:j for i,j in zip(unique_colours, np.arange(len(unique_colours)))}
for i in unique_colours:
    section[section==i] = color_map[i]

width, height = 1500, 1000
markers = [[636.8098159509204, 603.4958601655935, 672.6993865030674, 593.3762649494021], [902.9868982011025, 615.5567336628567, 843.8650306748466, 610.8555657773691], [561.2609204260139, 750.3661510917975, 558.5889570552147, 775.5289788408462]]
section = section.astype(np.uint8)

##you need to get all the pixel positions in section (something using meshgrid and np.arange)
##then transform each position (they would be scaled_x, and scaled_y in the function below)
##and then use the before and after position to place the previous pixel

resized_section = cv2.resize(section, dsize=(width, height))
triangulation = triangulate(width, height, markers)

new_x, new_y = transform_vec(triangulation, scaled_x, scaled_y)