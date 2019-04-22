#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:53:01 2019

@author: andrewgonzalez
"""

from __future__ import print_function
from sklearn.datasets import load_files
import numpy as np

train_dir = '/Users/andrewgonzalez/Desktop/fruits-360/Training'
test_dir = '/Users/andrewgonzalez/Desktop/fruits-360/Test'

num_classes = 95
batch_size = 35
epochs = 2


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files, targets, target_labels


# training data is just a file name of images. We need to convert them into pixel matrix
from keras.preprocessing.image import array_to_img, img_to_array, load_img


def convert_image_to_array(files):
    images_as_array = []
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array


################################################################################################

x_train, y_train, target_labels = load_dataset(train_dir)
x_test, y_test, _ = load_dataset(test_dir)
print('Loading complete!')
print('Training set size : ', x_train.shape[0])  # 48905
print('Testing set size: ', x_test.shape[0])  # 16421

no_of_classes = len(np.unique(y_train))
print(no_of_classes)  # 95 classes
print(y_train[0:10])

# target labels are numbers of corresponding to class label
# we need to change them to a vector of 95 elements since
# we have 95 classes
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, no_of_classes)
y_test = np_utils.to_categorical(y_test, no_of_classes)
print(y_train[0])

# No we have to divide the validation set into test and validation set
x_test, x_valid = x_test[7000:], x_test[:7000]
y_test, y_valid = y_test[7000:], y_test[:7000]
print('Validation X: ', x_valid.shape)
print('Validation y: ', y_valid.shape)
print('Test X: ', x_test.shape)
print('Test y: ', y_test.shape)

# Call the converting function to convert the files into array

x_train = np.array(convert_image_to_array(x_train))
print('Training set shape : ', x_train.shape)

x_valid = np.array(convert_image_to_array(x_valid))
print('Validation set shape : ', x_valid.shape)

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ', x_test.shape)

print('1st training image shape ', x_train[0].shape)

x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
x_test = x_test.astype('float32') / 255

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(100, 100, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

# Finally lets visualize the loss and accuracy wrt epochs

import matplotlib.pyplot as plt

plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()