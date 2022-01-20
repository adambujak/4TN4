import numpy as np
import cv2
import time

def detect_face(img):
    # convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # hue is represented as angle/2 in OpenCV
    # skin is in between 0 and 50 => 0 - 25 in OpenCV
    lower_hue = 1  # boost to 1 to remove some of red jacket in background
    upper_hue = 10 # lower to 10 to remove some of background

    # loop through every pixel
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j][0] <= lower_hue or img[i][j][0] >= upper_hue:
                # remove pixel outside above range
                img[i][j] = 0


    # convert back to BGR
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

# import image
img = cv2.imread('selfie.png')

# detect face
img = detect_face(img)

# display image
cv2.imshow('image', img)
# wait for 3 seconds
cv2.waitKey(3000)
# close image
cv2.destroyAllWindows()

# save face
cv2.imwrite('face.png', img)
