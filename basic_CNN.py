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
print('Training set shape : ',x_train.shape)

x_valid = np.array(convert_image_to_array(x_valid))
print('Validation set shape : ',x_valid.shape)

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

print('1st training image shape ',x_train[0].shape)

x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255

import matplotlib.pyplot as plt

fig = plt.figure(figsize =(30,5))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))


