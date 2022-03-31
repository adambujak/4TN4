import cv2
import matplotlib.pyplot as plt
import numpy as np

def find_matches(img1, img2):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    # exectue feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)

    return matches, keypoints_1, keypoints_2


def warp(img1, img2):
    matches, k1, k2 = find_matches(img1, img2)

    k1 = np.float32([kp.pt for kp in k1])
    k2 = np.float32([kp.pt for kp in k2])

    ptsA = np.float32([k1[m.queryIdx] for m in matches])
    ptsB = np.float32([k2[m.trainIdx] for m in matches])


    H, status = cv2.findHomography(ptsA,ptsB, cv2.RANSAC, 4)

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    out = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    out[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img2

    return out


# read images
img1 = cv2.imread('image2_1.jpg', 0)
img2 = cv2.imread('image2_2.jpg', 0)
img3 = cv2.imread('image2_3.jpg', 0)
img4 = cv2.imread('image2_4.jpg', 0)



top = warp(img1, img2)
bottom = warp(img3, img4)
out = warp(top, bottom)
cv2.imwrite('out3.png', out)






