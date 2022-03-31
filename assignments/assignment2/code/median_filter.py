import copy
myimage = [[3,5,8,4],
[9,1,2,9],
[4,6,7,3],
[3,8,5,4]]

mykernel = [[0,1,0],
[1,1,1],
[0,1,0]]



import statistics

def get_median_pixel(image, kernel, x, y):
    values = []
    for j in range(len(kernel)):
        for i in range(len(kernel[j])):
            image_val = 0
            iy = y + j - 1
            ix = x + i - 1
            if iy >= 0 and ix >= 0:
                try:
                    image_val = image[iy][ix]
                except:
                    pass

            if kernel[j][i] != 0:
                values += [kernel[j][i] * image_val]

    print(values)
    print(statistics.median(values))

    return statistics.median(values)


def median_filter(image, kernel):
    new_image = copy.deepcopy(image)
    for y in range(len(image)):
        for x in range(len(image[y])):
            new_image[y][x] = get_median_pixel(image, kernel, x, y)
    return new_image

print(myimage)
print(median_filter(myimage, mykernel))
