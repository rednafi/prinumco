import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage import io
from skimage import transform
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

model = load_model("./model/prinumco_mobilenet.h5")


def load(filename):

    image = Image.open(filename)
    np_image = np.array(image).astype("float32") / 255
    np_image = transform.resize(np_image, (96, 96, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


url = "./results/test.png"
image = load(url)

predict_matrix = model.predict(image)

print(np.argmax(predict_matrix))
