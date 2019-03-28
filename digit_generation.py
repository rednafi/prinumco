from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os 
from glob import glob
from tqdm import tqdm
import numpy as np
import Augmentor
import shutil
from Augmentor.Operations import Operation


# making directories
def directory_generator():
    for i in range(10):
        if not os.path.exists("train"):
            os.mkdir('train')

        if not os.path.exists('test'):
            os.mkdir('test')
            
        if not os.path.exists('train/'+ str(i)):
            os.mkdir('train/'+ str(i))
            
        if not os.path.exists('test/'+ str(i)):
            os.mkdir('test/' + str(i))
            
        else:
            pass


# checking the fonts
def check_fonts():

    """
    Make a directory named 'custom' in the '/usr/share/fonts/truetype' path and copy the bangla fonts there. 
    """
    path = '/usr/share/fonts/truetype/custom/'
    files = [f for f in glob(path + "**/*.ttf", recursive=True)]

    return files


# digit generation
def digit_generator(digit = '1', font_name = '/usr/share/fonts/truetype/custom/HindSiliguri-Regular.ttf',
                    font_size = 265, x_pos= 50, y_pos = -60, color = (255,255,255)):
    
    img = Image.new('RGB', (256, 256), color = color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font=font_name, size = font_size)
    d.text((x_pos, y_pos), digit, fill=(0, 0, 0), font=font)
    return img


# train data generation
def train_datagen(fonts, color_list, color_names):
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
    
    for idx, font_name in tqdm(enumerate(fonts)):
        for jdx, (digit_bn, digit_en) in enumerate(zip(digits_bns,digits_ens)):
            for color, color_name in zip(color_list, color_names): 
                try:
                    img = digit_generator(digit = digit_bn, font_name = font_name, color = color)
                    img.save('train/{}/{}_{}_{}.jpg'.format(digit_en,idx,jdx,color_name))
                except:
                    pass


# test data generation
def test_datagen(fonts):
    font_sizes = np.arange(150,200,1)
    digits_bns = "০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯".split()
    digits_ens = "0 1 2 3 4 5 6 7 8 9".split()

    for idx, font_name in tqdm(enumerate(fonts)):
        for jdx, font_size in enumerate(font_sizes):
            for kdx, (digit_bn, digit_en) in enumerate(zip(digits_bns,digits_ens)): 
                try:
                    img = digit_generator(digit = digit_bn, font_name = font_name, font_size=font_size)
                    img.save('test/{}/{}_{}_{}.jpg'.format(digit_en,idx,jdx,kdx))
                except:
                    pass



# image augmentation
class GaussianNoise(Operation):
    """
    The class `:class noise` is used to perfrom random noise on images passed
     to its :func:`perform_operation` function.
    """
    def __init__(self, probability, mean, std):
        Operation.__init__(self, probability)
        self.mean = mean
        self.std = std

    def perform_operation(self, images):
        def do(image):
            w, h = image.size
            c = len(image.getbands())
            noise = np.random.normal(self.mean, self.std, (h, w, c))/10
            return Image.fromarray(np.uint8(np.array(image) + noise ))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images

 
def augmentation(folder, noise, sample=100):
    p = Augmentor.Pipeline(folder)
    p.add_operation(noise)
    p.rotate90(probability=0.1)
    p.rotate270(probability=0.1)
    #p.crop_random(probability=1, percentage_area=0.9)
    p.zoom(probability=0.5, min_factor=1.01, max_factor=1.03)
    #p.crop_centre(probability = 0.5, percentage_area=0.5, randomise_percentage_area=True)

    p.flip_left_right(probability = 0.5)
    p.flip_random(probability = 0.5)
    p.flip_top_bottom(probability = 0.4)

    p.skew_tilt(probability = 0.5, magnitude = 0.5)
    p.skew_left_right(probability = 0.2, magnitude = 0.5)
    p.skew_top_bottom(probability = 0.6, magnitude = 0.5)
    p.skew_corner(probability = 0.1, magnitude = 1)
    p.skew(probability = 0.33, magnitude = 0.4)

    p.random_erasing(probability=0.3, rectangle_area=0.33)
    p.gaussian_distortion(probability = 0.3, grid_width = 5, grid_height = 5, magnitude = 5, corner = 'bell', method = 'in')
    p.random_brightness(probability = 0.8, min_factor = 0.5, max_factor = 1.5)
    # p.random_color(probability = 0.4, min_factor = 0.3, max_factor= 0.7 )
    p.random_contrast(probability= 0.4, min_factor = 0.3, max_factor = 0.7 )
    p.random_distortion(probability = 0.3, grid_width = 5, grid_height = 5, magnitude = 5)

    p.invert(probability = 0.5)
    p.histogram_equalisation(probability = 0.5)
    
    p.sample(sample, multi_threaded=True)


#moving the augmented images to the corresponding folders 
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in tqdm(os.listdir(src)):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

#removing the original augmented folder
def remove_output():
    shutil.rmtree('train/output')
    
# __main__
def main_func():
    directory_generator()
    fonts = check_fonts()
    digit_generator()
    color_list = [(255,255,255), (255, 255, 204)]
    color_names = ['white', 'yellow']
    train_datagen(fonts, color_list, color_names)
    test_datagen(fonts)
    augmentation('train/',noise = GaussianNoise(probability=0.3, mean = .09, std = 5), sample=1000)
    # src = 'train/output'
    # dst = 'train/'
    # copytree(src, dst)
    # remove_output()



main_func()



    



