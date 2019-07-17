import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
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

adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999)

base_dir = "./"
train_folder = base_dir + "dataset/train/"
test_folder = base_dir + "dataset/test/"

# keras model
input_shape = (96, 96, 3)
num_classes = 10

# defining the model
base_model = MobileNetV2(
    include_top=False, input_shape=input_shape, classes=num_classes
)

x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(1000, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])


# Data Generator
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_folder, target_size=(96, 96), batch_size=64, class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    test_folder, target_size=(96, 96), batch_size=64, class_mode="categorical"
)

model.fit_generator(
    train_generator,
    steps_per_epoch=3300,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=500,
)


# Saving the model
model.save("prinumco_mobilenet.h5")
