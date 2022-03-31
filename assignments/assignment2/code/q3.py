import numpy as np
import cv2
import time

# import image
img = cv2.imread('img3.png')

median = cv2.medianBlur(img,5)
gauss = cv2.GaussianBlur(img,(5,5),0)

#save img
cv2.imwrite('img3_median.png', median)
cv2.imwrite('img3_gauss.png', gauss)


