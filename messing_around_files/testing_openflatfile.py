base = r"../test_data\ttA_2877_NOP_atlasmaps\2877_NOP_tTA_lacZ_Xgal_s037_nl.flat"
import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt
import cv2

"""read flat file and write into an np array"""
np.set_printoptions(suppress=True)
with open(base, "rb") as f:
    # i dont know what b is, w and h are the width and height that we get from the
    # flat file header
    b, w, h = struct.unpack(">BII", f.read(9))
    # data is a one dimensional list of values
    # it has the shape width times height
    data = struct.unpack(">" + ("xBH"[b] * (w * h)), f.read(b * w * h))

    # convert flat file data into an array, previously data was a tuple
    imagedata = np.array(data)

    # create an empty image array in the right shape,
    image = np.zeros((h, w))
    # pallette = dict(zip(np.unique(data), np.random.randint(0,255,len(np.unique(data)))))

    # assign values from flat file into the image array
    for x in range(w):
        for y in range(h):
            image[y, x] = imagedata[x + y * w]

    image_arr = np.array(image)
    print(image_arr.shape)
    image_arr = cv2.resize(image_arr, (9848, 12784), interpolation=cv2.INTER_NEAREST)
    print(image_arr.shape)
    val, count = np.unique(image_arr, return_counts=True)
    print(list(zip(val, count)))

# show image with plt.imshow(image_arr)

"""assign label file values into image array"""

labelfile = pd.read_csv(r"annotation_volumes\allen2017_colours.csv")
#labelfile[]
print(list(zip(val,count)))


temp_copy = labelfile.loc[val.astype(int),:].copy()

temp_copy['region areas'] = count

temp_copy.to_csv('../outputs/region_areas_quick_test.csv',sep=';')

allen_id_image = np.zeros((h, w))  # create an empty image array

"""for ph in range(h):
    for pw in range(w):
        value_in_data_at_pixel = int(image_arr[ph,pw])
        allen_id_image[ph, pw] = labelfile.loc[value_in_data_at_pixel, 'idx']"""

"""for efficiency, vectorize instead of using the for loops above"""
coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))
values = image_arr[coordsx, coordsy]  # assign x,y coords from image_array into values
lbidx = labelfile["idx"].values
allen_id_image = lbidx[values.astype(int)]  # assign allen IDs into image array


"""count pixels for unique idx"""
unique_ids, counts = np.unique(allen_id_image, return_counts=True)

area_per_label = list(zip(unique_ids, counts))
# create a list of unique regions and pixel counts per region

df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "area_count"])
# create a pandas df with regions and pixel counts

print(df_area_per_label)
df_area_per_label.to_csv(
    "../outputs/s037_area_per_idx.csv", sep=";", na_rep="", index=False
)

# df_label_colours =pd.read_csv(label_colours, sep=",")
# find colours corresponding to each region ID and add to the pandas dataframe
