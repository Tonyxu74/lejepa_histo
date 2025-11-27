import numpy as np
from skimage.color import rgb2hed, hed2rgb
import random
from PIL import Image
import torch


def stain_augmentation(image, aug_mean=0, aug_std=0.035, threshold=40, threshold_upper=600):

    """
    Randomly augments staining of images by seperating them in to h and e (and d)
    channels and modifying their values. Aims to produce plausable stain variation
    used in custom augmentation
    :param image: arbitrary RGB image (3 channel array) expected to be 8-bit
    :param aug_mean: average value added to each stain, default setting is 0
    :param aug_std: standard deviation for random modifier, default value 0.035
    :param threshold: summative pixel value which will determine if a pixel will be ignored
    :param threshold_upper: summative pixel value which will determine if a pixel will be ignored
    :return: image - 8 bit RGB image with the same dimensions as the input image, with
    a modified stain
    """

    # creates a mask based on black and white areas of the original image so they are not augmented
    image = np.asarray(image, dtype=np.uint32)

    mask = (image.sum(axis=-1) > threshold) & (image.sum(axis=-1) < threshold_upper)

    ihc_hed = rgb2hed(image.astype(np.uint8))
    
    # get random stain modification, shape into (1,3) array
    hmod = random.normalvariate(aug_mean, aug_std)
    dmod = random.normalvariate(aug_mean, aug_std)
    emod = random.normalvariate(aug_mean, aug_std)
    mod_arr = np.array([[hmod, dmod, emod]])
    
    # use array broadcasting to add modification to areas within mask
    ihc_hed[mask, :] += mod_arr

    zdh = hed2rgb(ihc_hed)
    zdh_8bit = (zdh * 255).astype('uint8')

    return Image.fromarray(zdh_8bit)


class StainAug(torch.nn.Module):
    """
    Class wrapper for stain augmentation
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = stain_augmentation(x)

        return x
