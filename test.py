import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import transform
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from skimage import io

from tensorflow.keras.applications  import Xception#, VGG16, InceptionResNetV2, VGG19, InceptionV3

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

model = load_model('/media/redowan/New Volume/bangla_digit_generation/bangla_digits/primumco_validation_96.h5')

def load(url):
#    np_image = Image.open(filename)
    np_image = io.imread(url)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (75, 75, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

url = "https://i.imgur.com/SIFdnHa.png"
image = load(url)
import numpy as np

predict_matrix = model.predict(image)

print(np.argmax(predict_matrix.ravel()))

# print(np.argmax(model.predict(
# image)))
