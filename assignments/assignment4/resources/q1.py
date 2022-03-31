import numpy as np
import cv2

s=np.array([[0,0,0],[0,1,0],[1,1,1]])

def erode(image, size):
    size = size
    shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (size, size))
    return cv2.erode(working_image, element)

def dilate(image, size):
    size = size
    shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (size, size))
    return cv2.dilate(working_image, element)

image = cv2.imread('3.png', 0)

working_image = cv2.GaussianBlur(image, (51,51),cv2.BORDER_DEFAULT)

working_image = cv2.adaptiveThreshold(working_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

working_image = erode(working_image, 2)
working_image = dilate(working_image, 15)
working_image = erode(working_image, 35)
working_image = dilate(working_image, 70)
working_image = erode(working_image, 40)


cv2.imwrite('bitmask.png', working_image)
working_image = cv2.bitwise_and(working_image, image)

cv2.imwrite('license.png', working_image)


