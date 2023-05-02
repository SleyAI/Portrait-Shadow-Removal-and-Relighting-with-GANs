from __future__ import print_function
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


def adjust_gamma(image='/home/buens/Git/encoder4editing/inputs/images/00545.png', gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# construct the argument parse and parse the arguments
# load the original image
original = cv2.imread('/home/buens/Git/encoder4editing/inputs/images/00545.png')

# loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
    # ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
        continue
    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)



img = cv2.imread('/home/buens/Git/encoder4editing/inputs/images/00545.png')

R, G, B = cv2.split(img)

output1_R = cv2.equalizeHist(R)
output1_G = cv2.equalizeHist(G)
output1_B = cv2.equalizeHist(B)

equ = cv2.merge((output1_R, output1_G, output1_B))


# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cl1 = clahe.apply(output1_R)
cl2 = clahe.apply(output1_G)
cl3 = clahe.apply(output1_B)

cl = cv2.merge((cl1, cl2, cl3))

gan_processed = cv2.imread('/home/buens/Git/jan.buens/data/inversions/00545/00545_final_result/00545_410.jpg')
gan_processed = cv2.resize(gan_processed, (512,512))
res = np.hstack((img, equ, cl, adjusted, gan_processed))
cv2.imwrite('res.png', res)
