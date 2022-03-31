import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
import idx2numpy
import random
import pandas as pd
from sklearn import svm, metrics

imgsFilePath = "train-images.idx3-ubyte"
labelsFilePath = "train-labels.idx1-ubyte"
images = idx2numpy.convert_from_file(imgsFilePath) labels = idx2numpy.convert_from_file(labelsFilePath)
"""Load dataset and convert it to numpy array."""


