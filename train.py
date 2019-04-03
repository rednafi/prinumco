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

from tensorflow.keras.applications  import MobileNetV2#, VGG16, InceptionResNetV2, VGG19, InceptionV3

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999)

base_dir = './'
train_folder = base_dir + 'train/'
test_folder = base_dir + 'test/'

""" keras model """
input_shape = (96,96,3)
num_classes = 10

# defining the model
base_model = MobileNetV2(include_top=False, input_shape=input_shape, classes=num_classes)
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
        target_size=(96, 96),
<<<<<<< HEAD
        batch_size=64,
=======
        batch_size=32,
>>>>>>> 0cee80b54a53a59325bc7436c2122bbbe17a8612
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory( test_folder,
        target_size=(96, 96),
<<<<<<< HEAD
        batch_size=64,
=======
        batch_size=32,
>>>>>>> 0cee80b54a53a59325bc7436c2122bbbe17a8612
        class_mode='categorical')

model.fit_generator(
        train_generator,
<<<<<<< HEAD
        steps_per_epoch= 200,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=200)


""" Saving the model """
model.save('prinumco_v1.h5')
=======
        steps_per_epoch=20,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=10)


""" Saving the model """
#model.save('prinumco.h5')
>>>>>>> 0cee80b54a53a59325bc7436c2122bbbe17a8612

