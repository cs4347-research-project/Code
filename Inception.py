from __future__ import print_function
from sklearn.datasets import load_files
import numpy as np
from keras.utils import np_utils

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files, targets, target_labels

from keras.preprocessing.image import array_to_img, img_to_array, load_img
def convert_image_to_array(files):
    images_as_array = []
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array



train_dir = '/Users/andrewgonzalez/Desktop/fruits-360/Training'
test_dir = '/Users/andrewgonzalez/Desktop/fruits-360/Test'

num_classes = 95
batch_size = 35
epochs = 32


import keras
from keras.datasets import cifar10
#X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, y_train, target_labels = load_dataset(train_dir)
X_test, y_test, _ = load_dataset(test_dir)
print('Loading complete!')
print('Training set size : ', X_train.shape[0])  # 48905
print('Testing set size: ', X_test.shape[0])
no_of_classes = len(np.unique(y_train))


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

X_test, x_valid = X_test[7000:], X_test[:7000]
y_test, y_valid = y_test[7000:], y_test[:7000]

print('Validation X: ', x_valid.shape)
print('Validation y: ', y_valid.shape)
print('Test X: ', X_test.shape)
print('Test y: ', y_test.shape)

X_train = np.array(convert_image_to_array(X_train))
x_valid = np.array(convert_image_to_array(x_valid))
X_test = np.array(convert_image_to_array(X_test))
print('Training set shape : ', X_train.shape)
print('Validation set shape : ', x_valid.shape)
print('Test set shape : ', X_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
x_valid = x_valid.astype('float32') / 255



from keras.utils import np_utils

print('Do I get here?')
from keras.layers import Input
input_img = Input(shape = (100, 100, 3))

from keras.layers import Conv2D, MaxPooling2D
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)

from keras.layers import Conv2D, MaxPooling2D
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)

from keras.layers import Flatten, Dense
output = Flatten()(output)
out    = Dense(95, activation='softmax')(output)

from keras.models import Model
model = Model(inputs = input_img, outputs = out)
# print model.summary()

from keras.optimizers import SGD
epochs = 1
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

from keras.models import model_from_json
import os
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(os.getcwd(), 'model.h5'))

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Plot the accuracy and loss to help illustrate
import matplotlib.pyplot as plt
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()