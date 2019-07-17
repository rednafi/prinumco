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

from digit_generation_utils.digit_generation import digit_generator
from digit_generation_utils.digit_generation import test_datagen
from digit_generation_utils.digit_generation import train_datagen
from digit_generation_utils.directory_generation_check_font import check_fonts
from digit_generation_utils.directory_generation_check_font import directory_generator
from digit_generation_utils.image_augmentation import test_augmentation
from digit_generation_utils.image_augmentation import train_augmentation
from digit_generation_utils.mixing_aug_image_with_gen_image import copytree
from digit_generation_utils.mixing_aug_image_with_gen_image import remove_output

# function import


# __main__
def main():

    directory_generator()
    fonts = check_fonts()
    train_datagen(fonts, image_count=2320)
    test_datagen(fonts, image_count=2320)

    train_augmentation("dataset/train/", sample=200000)
    test_augmentation("dataset/test/", sample=28000)

    train_src = "dataset/train/output"
    train_dst = "dataset/train/"
    test_src = "dataset/test/output"
    test_dst = "dataset/test/"

    copytree(train_src, train_dst)
    copytree(test_src, test_dst)

    remove_output("dataset/train/output")
    remove_output("dataset/test/output")

    # train_test_image_binarize(train_folder= 'train/', test_folder= 'test/')


if __name__ == "__main__":
    main()
