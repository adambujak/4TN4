from scipy import signal
import numpy as np
import cv2

def scale(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

messi = cv2.imread('messi.jpg', 0)
messi = scale(messi, 2);
messi_copy = messi.copy()
messi = cv2.Canny(messi, 20, 120)

circle = cv2.imread('circle.bmp', 0)
for i in range(20):
    # start template at 70% scale and increase scale by 10% each time
    scale_factor = 0.7+0.1*i
    template = scale(circle, scale_factor)
    template = cv2.Canny(template, 20, 120)

    w, h = template.shape[::-1]
    img = messi.copy()
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc[0], max_loc[1]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(messi_copy, top_left, bottom_right, 255, 1)

cv2.imwrite('messi_4.png'.format(scale_factor), messi_copy)



