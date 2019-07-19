import os
import random
import shutil
from glob import glob

import Augmentor
import numpy as np
from Augmentor.Operations import Operation
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont
from tqdm import tqdm

from digit_generation import digit_generator
from digit_generation import test_datagen
from digit_generation import train_datagen
from directory_generation_check_font import check_fonts
from directory_generation_check_font import directory_generator
from image_augmentation import test_augmentation
from image_augmentation import train_augmentation
from mixing_aug_image_with_gen_image import copytree
from mixing_aug_image_with_gen_image import remove_output


# Desired image counts
TRAIN_IMAGE_COUNT = 2320
TEST_IMAGE_COUNT = 2320
TRAIN_AUGMENTED_COUNT = 200000
TEST_AUGMENTED_COUNT = 28000


# Image directories
train_src = "./dataset/train/output"
train_dst = "./dataset/train/"
test_src = "./dataset/test/output"
test_dst = "./dataset/test/"


# __main__
def main():

    directory_generator()
    fonts = check_fonts()
    train_datagen(fonts, image_count=TRAIN_IMAGE_COUNT)
    test_datagen(fonts, image_count=TEST_IMAGE_COUNT)

    train_augmentation("./dataset/train/", sample=TRAIN_AUGMENTED_COUNT)
    test_augmentation("./dataset/test/", sample=TEST_AUGMENTED_COUNT)

    copytree(train_src, train_dst)
    copytree(test_src, test_dst)

    remove_output("./dataset/train/output")
    remove_output("./dataset/test/output")

    # train_test_image_binarize(train_folder= 'train/', test_folder= 'test/')


if __name__ == "__main__":
    main()
