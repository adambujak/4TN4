from scipy import signal
import numpy as np
import cv2

def my_search(image, template):
    image = cv2.Canny(image, 20, 120)
    image = image - np.mean(image)

    template = cv2.Canny(template, 20, 120)
    template = template - np.mean(template)

    template = np.flipud(np.fliplr(template))
    corr = signal.convolve2d(image, template, boundary='symm', mode='full')

    corr *= (255.0/corr.max())
    cv2.imwrite('cross_correlate.png', corr)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

    w, h = template.shape[::-1]
    top_left = max_loc[0], max_loc[1]
    bottom_right = (top_left[0] - w, top_left[1] - h)
    cv2.rectangle(image, top_left, bottom_right, 255, 1)
    return image

messi = cv2.imread('messi.jpg', 0)
circle = cv2.imread('circle.bmp', 0)
messi = my_search(messi, circle)
cv2.imwrite('messi2.png', messi)
