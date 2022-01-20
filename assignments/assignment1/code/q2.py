import numpy as np
import cv2
import time

def gamma_correct(img, gamma):
    # loop through each pixel and apply gamma correction
    for i in range(len(img)):
        for j in range(len(img[i])):
            # loop through each BGR channel
            for k in range(len(img[i][j])):
                img[i][j][k] = (img[i][j][k]/255)**(1/gamma)*255
    return img


# import image
img = cv2.imread('img1.png')
# gamma correct
img = gamma_correct(img, 2)

# display image
cv2.imshow('image', img)
# wait for 3 seconds
cv2.waitKey(3000)
# close image
cv2.destroyAllWindows()
#save img
cv2.imwrite('gamma_correction.png', img)
