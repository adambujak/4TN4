import cv2 as cv
def get_error(x,y):
    error = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            error += (x[i][j] - y[i][j])**2
    return error

folder = "results/markers"

gt = cv.imread("{}/gt.png".format(folder), 0)
pe = cv.imread("{}/pe.png".format(folder), 0)
noise = cv.imread("{}/noise.png".format(folder), 0)
mf = cv.blur(noise, (3,3))


gt = gt[1000:1110, 1000:1110]

print("gt, gt: ", get_error(gt, gt))
print("gt, pe: ", get_error(gt, pe))
print("gt, mf: ", get_error(gt, mf))
print("gt, noise: ", get_error(gt, noise))
