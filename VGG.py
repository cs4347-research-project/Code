# importing the libraries


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob

# importing the libraries
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint

test_dir = '/Users/JBringmann/CS4347_Project/input/fruits/fruits-360/test-multiple_fruits'
training_dir = '/Users/JBringmann/CS4347_Project/input/fruits/fruits-360/Training'
validation_dir = '/Users/JBringmann/CS4347_Project/input/fruits/fruits-360/Test'

# useful for getting number of files
image_files = glob(training_dir + '/*/*.jp*g')
valid_image_files = glob(validation_dir + '/*/*.jp*g')

# getting the number of classes i.e. type of fruits
folders = glob(training_dir + '/*')
num_classes = len(folders)
print ('Total Classes = ' + str(num_classes))

checkpoint_path = "/Users/JBringmann/CS4347_Project/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
callbacks = [cp_callback]
IMAGE_SIZE = [64, 64]  

# loading the weights of VGG16 without the top layer. 
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer.

model = Model(inputs = vgg.input, outputs = x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2, 
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)
training_generator = training_datagen.flow_from_directory(
                                    training_dir, 
                                    target_size = IMAGE_SIZE, 
                                    batch_size = 200, 
                                    class_mode = 'categorical')

validation_datagen = ImageDataGenerator(
                                    rescale = 1./255, 
                                    preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
                                    validation_dir, 
                                    target_size = IMAGE_SIZE, 
                                    batch_size = 200, 
                                    class_mode = 'categorical')

test_datagen = ImageDataGenerator(
                                    rescale = 1./255, 
                                    preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
                                    validation_dir,
                                    target_size=IMAGE_SIZE,
                                    batch_size=200,
                                    class_mode= None)


model_history = model.fit_generator(training_generator,
                   steps_per_epoch = 10000,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results. 
                   epochs = 1,  # change this for better results  
                   callbacks=callbacks,
                   validation_data = validation_generator,
                   validation_steps = 3000)  # this should be equal to total number of images in validation set.

print ('Training Accuracy = ' + str(model_history.history['acc']))
print ('Validation Accuracy = ' + str(model_history.history['val_acc']))
