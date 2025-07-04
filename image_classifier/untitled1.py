# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IqGD0iP1C3lz7a3Q_-2bZ2Vg3r5Dyl2T
"""

!wget https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification?dataset_version_number=2

!unzip /content/intel-image-classification?dataset_version_number=2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

train_set=tf.keras.utils.image_dataset_from_directory(
    directory='/content/seg_train/seg_train',
    batch_size=16,
    image_size=(64, 64),
)

class_names=tf.constant(train_set.class_names)
print(class_names)

model=keras.Sequential(
    [
    keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),

    keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),

    keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),

    keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
    ]
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
    x=train_set,
    batch_size=None,
    epochs=10
)
model.save('model.h5')

from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('model.h5')

test_set=tf.keras.utils.image_dataset_from_directory(
    directory='/content/seg_test/seg_test',
    batch_size=32,
    image_size=(64, 64),
)
loss, accuracy = model.evaluate(test_set)

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

