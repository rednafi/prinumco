import Augmentor
from Augmentor.Operations import Operation
from PIL import Image, ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import random
import numpy as np 


# Bluring 
class Blur(Operation):
    # blur_type -> gaussian | box_blur | random
    # radius range -> (lower_range, upper_range)

    # if fixed radius is set a Float value, range won't be used 
    def __init__(self, probability, blur_type='gaussian', radius=(0, 1), fixed_radius=None):
        Operation.__init__(self, probability)
        self.blur_type = blur_type
        self.radius = radius
        self.fixed_radius = fixed_radius

    def perform_operation(self, images):
        def do(image):
            # Choose blur type 
            if self.blur_type != 'random':
                blur_filter = ImageFilter.GaussianBlur if self.blur_type == 'gaussian' else ImageFilter.BoxBlur
            else:
                coin_toss = random.choice([ 'gaussian', 'box_blur' ])
                blur_filter = ImageFilter.GaussianBlur if coin_toss == 'gaussian' else ImageFilter.BoxBlur
            # Choose radius 
            if self.fixed_radius == None:
                assert len(self.radius) == 2
                radius = np.random.uniform(low=self.radius[0], high=self.radius[1])
            else:
                radius = self.fixed_radius
            return image.filter(blur_filter(radius=radius))

        augmented_images = []
        
        for image in images:
            augmented_images.append(do(image))

        return augmented_images


# Filter op
class Filters(Operation):
    # Filter types ->  mode, median, max, min, random
    def __init__(self, probability, filter_type='mean', sizes = None, size=3):
        Operation.__init__(self, probability)
        self.filter_type = filter_type
        self.size = size
        self.sizes = sizes
        self.filters = {
            'median' : ImageFilter.MedianFilter,
            'min' : ImageFilter.MinFilter,
            'max' : ImageFilter.MaxFilter,
            'mode' : ImageFilter.ModeFilter
        }
    
    def perform_operation(self, images):
        def do(image):
            if self.sizes == None:
                size = self.size
            else:
                size = random.choice(self.sizes)
            
            if self.filter_type == 'random':
                filter_type = random.choice(['median', 'max', 'min', 'mode'])
            else:
                filter_type = self.filter_type

            return image.filter(self.filters[filter_type](size=size))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
    


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
            noise = np.random.normal(self.mean, self.std, (h, w, c))
            return Image.fromarray(np.uint8(np.asarray(image) + 0.003*noise ))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images

# Add noise operation
noise = GaussianNoise(probability=0.9, mean = 0, std = 20.0 )
_filter = Filters(probability=0.8, filter_type='random', size=5)
blur = Blur(probability=0.7, blur_type='random', radius=(0, 100), fixed_radius=3)


def augmentation(folder, sample=100):
    p = Augmentor.Pipeline(folder)
    p.add_operation(_filter)
    p.add_operation(blur)
    p.add_operation(noise)

    #p.black_and_white(probability = 1, threshold = 128)
    p.rotate(probability = 0.3, max_left_rotation = 25, max_right_rotation = 25)
    p.rotate90(probability = 0.005)
    p.rotate270(probability = 0.005)
    p.crop_random(probability=0.3, percentage_area = 0.9)
    p.zoom(probability=0.1, min_factor=1.01, max_factor=1.03)


    p.skew_tilt(probability = 0.03, magnitude = 1)
    p.skew_left_right(probability = 0.03, magnitude = 1)
    p.skew_top_bottom(probability = 0.02, magnitude = 1)
    p.skew_corner(probability = 0.03, magnitude = 1)
    p.skew(probability = 0.03, magnitude = 1)

    p.random_erasing(probability=0.01, rectangle_area=0.11)
    p.random_brightness(probability = 0.8, min_factor = 0.5, max_factor = 1.5)
    p.random_distortion(probability = 0.01, grid_width = 2, grid_height = 2, magnitude = 1)

    p.invert(probability = 0.09)
    p.resize(probability = 1, width = 256, height = 256)
    p.sample(sample, multi_threaded=True)

