from PIL import Image, ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import numpy as np
import os


# digit generation
def digit_generator(digit = '1', font_name = '/usr/share/fonts/truetype/custom/HindSiliguri-Regular.ttf',
                    font_size = 265, x_pos= 50, y_pos = -60, color = (255,255,255)):
    
    img = Image.new('RGB', (256, 256), color = color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font=font_name, size = font_size)
    d.text((x_pos, y_pos), digit, fill=(0, 0, 0), font=font)
    return img


# train data generation
def train_datagen(fonts, color_list, color_names, image_count = 100):
    """
    color_list is a list of tuples like (255,255,255) and color_names represents the corresponding names.
    ------------------------------------------------------------------------------------------------------
    Example:
    color_list = [(255,255,255), (255, 255, 204), (255, 153, 102), (102, 255, 51), (0, 0, 255), (255, 0, 102)]
    color_names = color_names = ['white', 'yellow', 'orange', 'green', 'blue', 'red']
    ------------------------------------------------------------------------------------------------------
    """
    digits_bns = "০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯".split()
    digits_ens = "0 1 2 3 4 5 6 7 8 9".split()
    
    if len(os.listdir('train/0')) == 0:
        print('Generating training images...')
        img_cnt = 0
        for idx, font_name in tqdm(enumerate(fonts)):
            for jdx, (digit_bn, digit_en) in enumerate(zip(digits_bns,digits_ens)):
                for color, color_name in zip(color_list, color_names): 
                    try:
                        img = digit_generator(digit = digit_bn, font_name = font_name, color = color)
                        img_cnt += 1
                        if img_cnt <= image_count: 
                            img.save('train/{}/{}_{}_{}.jpg'.format(digit_en,idx,jdx,color_name))
                    except:
                        pass

    else:
        print('Directory is not empty: Not generating training images')


# test data generation
def test_datagen(fonts, image_count = 100):
    font_sizes = np.arange(150,200,1)
    digits_bns = "০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯".split()
    digits_ens = "0 1 2 3 4 5 6 7 8 9".split()
    
    if len(os.listdir('test/0')) ==0:
        print('Generating test images...')
        img_cnt = 0
        for idx, font_name in tqdm(enumerate(fonts)):
            for jdx, font_size in enumerate(font_sizes):
                for kdx, (digit_bn, digit_en) in enumerate(zip(digits_bns,digits_ens)): 
                    try:
                        img = digit_generator(digit = digit_bn, font_name = font_name, font_size=font_size)
                        img_cnt += 1
                        if img_cnt <= image_count:
                            img.save('test/{}/{}_{}_{}_{}.jpg'.format(digit_en,idx,jdx,kdx, font_name.split('.ttf')[0].split('/')[-1]))
                    except:
                        pass

    else:
        print('Directory is not empty: Not generating test images')





    



