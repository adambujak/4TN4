import cv2
import matplotlib.pyplot as plt

# read images
img1 = cv2.imread('image1_1.jpg', 0)
img2 = cv2.imread('image1_2.jpg', 0)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# exectue feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)

matches = list(matches)
# sort matches by distance
for i in range(len(matches)):
    for j in range(len(matches)):
        if matches[j].distance > matches[i].distance:
            temp = matches[j]
            matches[j] = matches[i]
            matches[i] = temp

# only use best 30 matches
matches = matches[0:30]

out = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img2, flags=2)

cv2.imwrite('image_1_matches.png', out)
