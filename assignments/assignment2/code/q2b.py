import numpy as np
import cv2
import time

def apply_mask_in_frequency_domain(img, mask):
    img_fft = np.fft.fft2(img)
    img_shift = np.fft.fftshift(img_fft)

    masked_fft = img_shift * mask
    new_img = np.fft.ifftshift(masked_fft)

    new_img = np.fft.ifft2(new_img)
    new_img = np.abs(new_img)
    return new_img


# import messi image
messi = cv2.imread('messi.jpg', 0)
# import ronaldo image
ronaldo = cv2.imread('ronaldo.jpg', 0)

image = np.zeros((len(messi), len(messi[0])), np.uint8)
center = (int(len(messi[0])/2), int(len(messi)/2))
circle = cv2.circle(image, center, 30, 1, -1)

messi_new = np.multiply(apply_mask_in_frequency_domain(messi, circle), 0.5)
inv_circle = np.divide(cv2.bitwise_not(circle), 255)
ronaldo_new = np.multiply(apply_mask_in_frequency_domain(ronaldo, inv_circle), 0.5)

#save img
cv2.imwrite('messi_circle.png', cv2.add(messi_new, ronaldo_new))

