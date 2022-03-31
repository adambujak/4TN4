from scipy import signal
import numpy as np
import cv2

messi = cv2.imread('messi.jpg', 0)
messi = cv2.Canny(messi, 20, 120)
circle = cv2.imread('circle.bmp', 0)
circle = cv2.Canny(circle, 20, 120)

w, h = circle.shape[::-1]
img = messi.copy()
res = cv2.matchTemplate(img,circle,cv2.TM_CCOEFF_NORMED)

res *= (255.0/res.max())
cv2.imwrite('cv2_template_match.png', res)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc[0], max_loc[1]
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(messi, top_left, bottom_right, 255, 1)
cv2.imwrite('messi3.png', messi)



