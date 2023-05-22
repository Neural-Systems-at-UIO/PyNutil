base= r"../test_data\ttA_2877_NOP_atlasmaps\2877_NOP_tTA_lacZ_Xgal_s037_nl.flat"
import numpy as np
import struct
with open(base,"rb") as f:
    b,w,h=struct.unpack(">BII",f.read(9))
    data=struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))
    
from PIL import Image
import random
image= Image.new("RGB",(w,h))
for y in range(h):
    for x in range(w):
    image.putpixel((x,y),palette[d])

image_arr = np.array(image)

image_arr.min()