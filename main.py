from PIL import Image, ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import os 
from glob import glob
import random
from tqdm import tqdm
import numpy as np
import Augmentor
import shutil
from Augmentor.Operations import Operation


# function import 
from digit_generation_utils.directory_generation_check_font import directory_generator, check_fonts
from digit_generation_utils.digit_generation import digit_generator, train_datagen, test_datagen
from digit_generation_utils.image_augmentation import augmentation
from digit_generation_utils.mixing_aug_image_with_gen_image import copytree, remove_output 
from digit_generation_utils.preprocess import train_test_image_binarize

# __main__
def main_func():
    directory_generator()
    fonts = check_fonts()
    train_datagen(fonts, image_count=100)
    test_datagen(fonts, image_count=30)
    augmentation('train/', sample=100)
    src = 'train/output'
    dst = 'train/'
    copytree(src, dst)
    remove_output('train/output')
    train_test_image_binarize(train_folder= 'train/', test_folder= 'test/')



main_func()
