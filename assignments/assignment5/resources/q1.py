import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

img = cv2.imread('chess.png')
height, width, channels = img.shape

temp = np.full((height + 200, width + 200, 3), 255, dtype='uint8')
temp[100:height+100, 100:width+100] = img

#plt.imshow(temp)
#plt.show()

img = temp


img_cpy = np.copy(img)


# x, y
BOTTOM_LEFT = (66, 358)
BOTTOM_RIGHT = (475, 801)
TOP_LEFT = (511, 170)
TOP_RIGHT = (916, 408)

IMAGE_WIDTH = int(max(get_distance(BOTTOM_LEFT, BOTTOM_RIGHT), get_distance(TOP_LEFT, TOP_RIGHT)))
IMAGE_HEIGHT = int(max(get_distance(BOTTOM_LEFT, TOP_LEFT), get_distance(BOTTOM_RIGHT, TOP_RIGHT)))

start_point_arr = [TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT, TOP_RIGHT]
start_points = np.float32(start_point_arr)
end_points = np.float32([[0,0], [0, IMAGE_HEIGHT], [IMAGE_WIDTH, IMAGE_HEIGHT], [IMAGE_WIDTH, 0]])

for pt in start_point_arr:
    cv2.circle(img_cpy, pt, 3, (255, 155, 100), cv2.FILLED)

cv2.imwrite('chesspts.png', img_cpy)

T = cv2.getPerspectiveTransform(start_points, end_points)

img = cv2.warpPerspective(img, T, (IMAGE_WIDTH, IMAGE_HEIGHT), flags=cv2.INTER_LINEAR)


cv2.imwrite('new_chess.png',img)


