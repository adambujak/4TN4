import numpy as np
import array_to_latex as a2l

image = np.array([
[0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,1,1,1,1,0,0],
[0,0,0,0,1,1,1,1,1,1,0,0],
[0,0,0,0,0,1,1,1,1,1,0,0],
[0,0,0,0,0,0,1,1,1,1,0,0],
[0,0,1,0,0,0,0,1,1,1,0,0],
[0,0,0,1,0,0,0,0,1,1,0,0],
[0,0,0,0,1,0,0,0,0,1,0,0],
[0,0,0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0]])


s1 = np.array([
[0,1,0],
[0,1,1],
[0,1,0]])

s2 = np.array([
[1,0,0],
[0,1,0],
[0,0,1]])

s3 = np.array([
[1,1,1],
[1,1,1],
[1,1,1]])

def dilate(image, s):
    x,y = image.shape
    output = np.full((x,y), 0)
    for i in range(1, x-1):
        for j in range(1, y-1):

            temp = 0
            for k in range(3):
                for l in range(3):
                    if s[k][l] == 0:
                        continue
                    temp = temp or image[i-1+k][j-1+l]

            output[i][j] = temp

    return output



def erode(image, s):
    x,y = image.shape
    output = np.full((x,y), 0)
    for i in range(1, x-1):
        for j in range(1, y-1):

            temp = 1
            for k in range(3):
                for l in range(3):
                    if s[k][l] == 0:
                        continue
                    temp = temp and image[i-1+k][j-1+l]

            output[i][j] = temp

    return output


output = erode(image, s3)
print('erode')
a2l.to_ltx(output, frmt = '{:d}')
print('dilate')
output = dilate(output, s3)
a2l.to_ltx(output, frmt = '{:d}')
