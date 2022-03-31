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

def get_rgb_histograms(img):
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

        bins = np.arange(0,256,1)
        histvals = []
        for histval in hist:
            histvals += [int(histval[0])]
        plot_histogram(bins, histvals, color, color)

def apply_histogram_equalization_rgb(img, clahe: bool):
    chans = cv2.split(img)
    new_chans = []
    for chan in chans:
        if clahe:
            clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
            chan = clahe.apply(chan)
        else:
            chan = cv2.equalizeHist(chan)
        new_chans += [chan]
    img = cv2.merge(new_chans)
    return img

USE_CLAHE = True
# import image
img = cv2.imread('img4.png')

get_rgb_histograms(img)
img = apply_histogram_equalization_rgb(img, USE_CLAHE)
get_rgb_histograms(img)

img_name = "img4_rgb_he"
if USE_CLAHE:
    img_name += "CLAHE"
img_name += ".png"

#save img
cv2.imwrite(img_name, img)
