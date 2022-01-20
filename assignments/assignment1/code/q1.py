import numpy as np
import cv2
import time

def rotate(img, degree):
    # if degree is not 0, 90, 180, 270 throw exception
    if degree % 90 != 0:
        raise ValueError('invalid angle')

    if degree == 0:
        return img

    # rotation code
    rot = int(degree / 90) - 1

    newimg = []
    # rotate image
    return cv2.rotate(img, rot)

# import image
img = cv2.imread('img1.png', 0)

# get rotation angle
rotangle = int(input("enter rotation angle: "))
img = rotate(img, rotangle)

# display image
cv2.imshow('image', img)
# wait for 3 seconds
cv2.waitKey(3000)
# close image
cv2.destroyAllWindows()
