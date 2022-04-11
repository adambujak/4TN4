from joblib import load
from sklearn.svm import SVR
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# load computed model
regressor = load('regressor.joblib')

def estimate_pixel(X):
    global regressor
    pixel = regressor.predict([X]);
    return pixel

# read image
def get_images(directory):
    # read noisy image
    # read ground truth - henceforth GT

    # get files in directory
    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    noisy_image_name = None
    gt_image_name = None
    for file in files:
        if "NOISY" in file:
            noisy_image_name = "{}/{}".format(directory, file)

        if "GT"  in file:
            gt_image_name = "{}/{}".format(directory, file)

    if noisy_image_name is None or gt_image_name is None:
        raise ValueError('images not found')

    noisy_image = cv.imread(noisy_image_name, 0)
    gt_image = cv.imread(gt_image_name, 0)

    return noisy_image, gt_image

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged

def get_image_edges(image):
    # blur image to reduce noise
    blur = cv.GaussianBlur(image, (9,9), 0)
    # perform canny edge detection
    edges = auto_canny(blur)

    return edges

def extract_training_point(noisy_image, gt_image, edge_image, x, y):
    noise_block = noisy_image[y-1:y+2, x-1:x+2]
    edge_block = edge_image[y-1:y+2, x-1:x+2]

    noise_block = noise_block.flatten()
    edge_block = edge_block.flatten()
    expected_value = gt_image[y][x]

    X = np.append(noise_block, edge_block)
    y = expected_value
    return X, y

def generate_reduced_noise_image(noisy_image, gt_image, edge_image):
    new_image = noisy_image.copy()

    for j in range(1, len(noisy_image)-1, 1):
        for i in range(1, len(noisy_image[j])-1, 1):
            tX, ty = extract_training_point(noisy_image, gt_image, edge_image, i, j)
            pixel_estimate = estimate_pixel(tX)
            new_image[j][i] = pixel_estimate
    return new_image

def reduce_noise(directory):
    print("reducing noise in image from {}".format(directory))
    noisy_image, gt_image = get_images(directory)
    edge_image = get_image_edges(noisy_image)
    new_image = generate_reduced_noise_image(noisy_image, gt_image, edge_image)
    cv.imwrite("new_image.png", new_image)
    cv.imwrite("noisy_image.png", noisy_image)
    cv.imwrite("ground_image.png", gt_image)

def get_image_directory_list():
    directory = "training_data/Data"
    directories = [f for f in listdir(directory) if not isfile(join(directory, f))]
    directories = ['{}/{}'.format(directory, path) for path in directories]
    return directories

def run():
    training_directories = get_image_directory_list()
    training_directories.sort()

    for path in training_directories:
        reduce_noise(path)
        break


run()
