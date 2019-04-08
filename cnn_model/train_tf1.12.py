import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import os 
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout

from keras.applications.mobilenet_v2  import MobileNetV2

from keras.optimizers import Adam, RMSprop, SGD
adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999)

base_dir = './'
train_folder = base_dir + 'dataset/train/'
test_folder = base_dir + 'dataset/test/'

""" keras model """
input_shape = (96,96,3)
num_classes = 10

# defining the model
base_model = MobileNetV2(include_top=False, input_shape=input_shape, classes=num_classes)
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(1000, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


""" Data Generator """

train_datagen = ImageDataGenerator(rescale=1./255,
                
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_folder,
        target_size=(96, 96),
        batch_size=64,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory( test_folder,
        target_size=(96, 96),
        batch_size=64,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=3300,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=500)


""" Saving the model """
model.save('prinumco_mobilenet.h5')
