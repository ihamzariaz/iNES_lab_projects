from __future__ import print_function
import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import os
import math
import numpy
import time
#from PIL import Image

batch_size = 128
num_classes = 1000
epochs = 5
data_augmentation = False

label_counter = 0

training_images = []
training_labels = []

for subdir, dirs, files in os.walk('./tiny-imagenet-200/train/'):
    for folder in dirs:
        for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
            for file in folder_files:
                training_images.append(os.path.join(folder_subdir, file))
                training_labels.append(label_counter)

        label_counter = label_counter + 1

nice_n = math.floor(len(training_images) / batch_size) * batch_size

print(nice_n)
print(len(training_images))
print(len(training_labels))

import random
perm = list(range(len(training_images)))
random.shuffle(perm)
training_images = [training_images[index] for index in perm]
training_labels = [training_labels[index] for index in perm]

print("Data is ready...")

def get_batch():
    index = 1

    global current_index

    B = numpy.zeros(shape=(batch_size, 64, 64, 3))
    L = numpy.zeros(shape=(batch_size))

    while index < batch_size:
        try:
            img = load_img(training_images[current_index])
            B[index] = img_to_array(img)
            B[index] /= 255

            L[index] = training_labels[current_index]

            index = index + 1
            current_index = current_index + 1
        except:
            print("Ignore image {}".format(training_images[current_index]))
            current_index = current_index + 1

    return B, keras.utils.to_categorical(L, num_classes)

model = Sequential()

# conv1
model.add(Conv2D(16, (3, 3), padding='same', input_shape= (64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv2
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv3
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv4
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv5
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# fc1
model.add(Dense(2048))
model.add(Activation('relu'))

# fc2
model.add(Dense(num_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

for i in range(0, epochs):
    current_index = 0

    while current_index + batch_size < len(training_images):
        start_time = time.time()

        b, l = get_batch()

        loss, accuracy = model.train_on_batch(b, l)
        end_time = time.time()

        print('batch {}/{} loss: {} accuracy: {} time: {}ms'.format(int(current_index / batch_size), int(nice_n / batch_size), loss, accuracy, 1000 * (end_time - start_time)), flush=True)

    print('epoch {}/{}'.format(i, epochs))

current_index = 0
loss = 0.0
acc = 0.0

while current_index + batch_size < len(training_images):
    b, l = get_batch()

    score = model.test_on_batch(b, l)
    print('Test batch score:', score[0])
    print('Test batch accuracy:', score[1], flush = True)

    loss += score[0]
    acc += score[1]

loss = loss / int(nice_n / batch_size)
acc = acc / int(nice_n / batch_size)

print('Test score:', loss)
print('Test accuracy:', acc)

###### To test ###########
from scipy.misc import imread, imresize
img = imread('test7.jpg', mode='RGB')
img = imresize(img, [64, 64])
img = img.reshape((1, 64, 64, 3))
pred=model.predict(img)