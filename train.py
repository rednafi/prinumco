import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout

from tensorflow.keras.applications  import Xception#, VGG16, InceptionResNetV2, VGG19, InceptionV3

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999)

base_dir = './'
train_folder = base_dir + 'train/'
test_folder = base_dir + 'test/'

""" keras model """
input_shape = (75,75,3)
num_classes = 10

# defining the model
base_model = Xception(include_top=False, input_shape=input_shape, classes=num_classes)
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(1000, activation='relu')(x)
x = Dense(512, activation='relu')(x)
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
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory( test_folder,
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=6700,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=910)


""" Saving the model """
model.save('prinumco.h5')

