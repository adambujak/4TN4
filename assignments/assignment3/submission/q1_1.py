import numpy as np
import cv2
from matplotlib import pyplot as plt

#read image
img = cv2.imread('lp.jpg', 0)

blurred_small = cv2.GaussianBlur(img, (3, 3), 0)
blurred_big = cv2.GaussianBlur(img, (7, 7), 0)

output = (cv2.subtract(blurred_small, blurred_big))
cv2.imwrite('q1_dog.png', output)

output = cv2.Canny(img, 20, 120)
cv2.imwrite('q1_canny.png', output)


