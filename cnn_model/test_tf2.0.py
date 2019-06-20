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
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dropout,
)
from skimage import io
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


model = load_model("prinumco_mobilenet.h5")


def load(filename):

    image = Image.open(filename)
    np_image = np.array(image).astype("float32") / 255
    np_image = transform.resize(np_image, (96, 96, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


url = "test.png"
image = load(url)

predict_matrix = model.predict(image)

print(np.argmax(predict_matrix))

