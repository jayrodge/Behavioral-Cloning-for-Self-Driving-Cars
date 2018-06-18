import os
import csv
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Activation
from keras.layers import Lambda
from keras import optimizers
from keras.layers.convolutional import MaxPooling2D,Convolution2D,Cropping2D,Conv2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=30):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # name = './IMG/'+batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name)
                # center_angle = float(batch_sample[3])
                # images.append(center_image)
                # angles.append(center_angle)
                center = './data/IMG/'+batch_sample[0].split('/')[-1]
                left = './data/IMG/'+batch_sample[1].split('/')[-1]
                right = './data/IMG/'+batch_sample[2].split('/')[-1]
                image_center = cv2.imread(center)
                image_left = cv2.imread(left)
                image_right = cv2.imread(right)
                angle = float(batch_sample[3])
                images.append([image_center, image_left, image_right])
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=30)
validation_generator = generator(validation_samples, batch_size=30)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row,col ,ch),
        output_shape=(row, col,ch)))
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(6, 5, 5,activation='relu'))
print('Convolute')
# model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(60))
model.add(Dropout(0.2))
# model.add(Dense(30))
# model.add(Dropout(0.3))
model.add(Dense(10))
#model.add(Dropout(0.2))
model.add(Dense(1))
print('Model Done')
#sgd = optimizers.SGD(lr=0.01)
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=3)