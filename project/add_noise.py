import numpy as np
import cv2
from imgaug import augmenters as iaa


def add_noise(image_name, image_extension):
    im_arr = cv2.imread("{}.{}".format(image_name, image_extension))

    #im_arr = np.asarray(im)

    # gaussian noise
    # aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)

    # poisson noise
    aug = iaa.AdditivePoissonNoise(lam=30.0, per_channel=True)

    # salt and pepper noise
    #aug = iaa.SaltAndPepper(p=0.05)

    im_arr = aug.augment_image(im_arr)

    cv2.imwrite('{}_wn.{}'.format(image_name, image_extension), im_arr)

add_noise("astro_train/1", "jpg")
#add_noise("portrait", "jpeg")
