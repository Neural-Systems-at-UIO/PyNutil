from glob import glob
import cv2

images = glob("/home/harryc/github/PyNutil/tests/test_data/blank_test/segmentations/*")
for image in images:
    im = cv2.imread(image)
    im[:] = 255
    cv2.imwrite(image, im)