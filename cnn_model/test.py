import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage import io, transform
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

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
