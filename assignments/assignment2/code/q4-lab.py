import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_histogram(bins, values, channel_name, color):
    fig = plt.figure()
    plt.ylabel('frequency')
    plt.xlabel('bin')
    plt.xlim([0, 256])
    plt.title("{} channel histogram".format(channel_name))
    plt.bar(bins, values, color=color)
    plt.show()

def get_lab_histogram(img):
    l_channel,a_channel,b_channel = cv2.split(img)
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])

    bins = np.arange(0,256,1)
    histvals = []
    for histval in hist:
        histvals += [int(histval[0])]
    plot_histogram(bins, histvals, 'L', None)

def apply_histogram_equalization_lab(img, clahe: bool):
    L, A, B = cv2.split(img)
    if clahe:
        clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
        Le = clahe.apply(L)
    else:
        Le = cv2.equalizeHist(L)
    img = cv2.merge((Le, A, B))
    return img

USE_CLAHE = True
# import image
img = cv2.imread('img4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# show histogram of l channel
get_lab_histogram(img)
# apply histogram equalization on l channel
img = apply_histogram_equalization_lab(img, USE_CLAHE)
# show histogram of l channel
get_lab_histogram(img)
img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

img_name = "img4_lab_he"
if USE_CLAHE:
    img_name += "CLAHE"
img_name += ".png"

#save img
cv2.imwrite(img_name, img)
