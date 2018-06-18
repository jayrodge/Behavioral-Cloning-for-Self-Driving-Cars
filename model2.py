import csv
import os
import cv2
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# GLOBALS
csv_path = './data/driving_log.csv'
img_dir_path = './data/IMG'
correction = 0.2

def processImage(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def dataGenerator(samples, batch_size=64):
	"""
	Generator for data batches containing images for input and steering angles
	for output.
	"""
	num_samples = len(samples)

	while True:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				# generate path and read images
				img_path_c = os.path.join(img_dir_path, os.path.basename(batch_sample[0]))
				img_path_l = os.path.join(img_dir_path, os.path.basename(batch_sample[1]))
				img_path_r = os.path.join(img_dir_path, os.path.basename(batch_sample[2]))

				image_center = processImage(cv2.imread(img_path_c))
				image_left = processImage(cv2.imread(img_path_l))
				image_right = processImage(cv2.imread(img_path_r))

				# read steering angles and correct for camera location
				steering_center = float(batch_sample[3])
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				images.append(image_center)
				images.append(image_left)
				images.append(image_right)
				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)

			# augment images + angles by flipping
			augmented_images = []
			augmented_angles = []
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_images.append(cv2.flip(image, 1))
				augmented_angles.append(angle)
				augmented_angles.append(-angle)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def train(train_generator, valid_generator, num_train, num_valid):
	"""
	Trains the model using the data generators. The model is built with Keras
	using a modified LeNet architecture based on the network used for the
	traffic sign classifier project.
	"""
	model = Sequential()

	# pre-processing
	model.add(Cropping2D(cropping=((65, 20), (0, 0)), input_shape=(160,320,3)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))

	# convolutional layers
	model.add(Convolution2D(8, 1, 1, activation='relu'))
	model.add(Convolution2D(16, 5, 5, activation='relu'))
	model.add(MaxPooling2D(border_mode='same'))
	model.add(Convolution2D(32, 5, 5, activation='relu'))
	model.add(MaxPooling2D(border_mode='same'))

	# fully connected layers
	model.add(Flatten())
	model.add(Dropout(0.7))
	model.add(Dense(120, activation='relu'))
	model.add(Dense(84, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_generator,
						samples_per_epoch=num_train,
						validation_data=valid_generator,
						nb_val_samples=num_valid,
						nb_epoch=3)

	model.save('model.h5')

def main():
	samples = []
	with open(csv_path, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			samples.append(row)

	# create generators
	train_samples, valid_samples = train_test_split(samples, test_size=0.2)
	train_generator = dataGenerator(train_samples, 64)
	valid_generator = dataGenerator(valid_samples, 64)

	# train model
	train(train_generator, valid_generator, len(train_samples), len(valid_samples))

if __name__ == '__main__':
    gc.collect()
    main()

# import csv
# import cv2
# import numpy as np
# import keras
# from keras.models import Sequential, load_model
# from keras.layers import Cropping2D
# from keras.layers.convolutional import Convolution2D
# from keras.layers.core import Dense, Activation, Flatten, Lambda
# from keras.layers.pooling import MaxPooling2D
# import sklearn
# from random import shuffle
# from sklearn.model_selection import train_test_split
# import copy
# import argparse

# # argument format: python model.py -f foldername -e 5 -l 5e-4

# def main():
#     parser = argparse.ArgumentParser(description='Train Keras Model')
#     parser.add_argument('-f', '--foldernames', nargs='+', type=str)
#     parser.add_argument('-l', '--lr_late', default = 5e-4, type=float)
#     parser.add_argument('-e', '--epcohs', default = 5, type=int)
#     args = parser.parse_args()
#     # **Caution** data was collected on Windows with the path format:
#     #\\data\\foldernames\\IMG\\center_2017_10_26_01_32_48_251.jpg'.
#     # But the training was done on linux, so '\\' has to be replaced by '/' for general use.
#     foldernames = args.foldernames
#     samples = []
#     for foldername in foldernames:
#         print('Including data from folder: {}'.format(foldername))
#         with open('data/' + foldername + '/driving_log.csv') as csvfile:
#             reader = csv.reader(csvfile)
#             for line in reader:
#                 samples.append(line)


#     train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#     print('Train Sample Size = {}'.format(len(train_samples)))
#     print('Valid Sample Size = {}'.format(len(validation_samples)))

#     # Define Data Generator with augmentation function inclduing center, right, left images
#     # and their flipped ones. When shuffing, I have two different shuffle lists for original and
#     # fliiped images so it's better for model generalization
#     def generator(samples, batch_size=32):
#         num_samples = len(samples)
#         while 1: # Loop forever so the generator never terminates
#             samples_flipped = copy.copy(samples)
#             shuffle(samples)
#             shuffle(samples_flipped)
#             for offset in range(0, num_samples, batch_size):
#                 batch_samples = samples[offset:offset+batch_size]
#                 batch_samples_flipped = samples_flipped[offset:offset+batch_size]
#                 images = []
#                 angles = []
#                 correction_left = 0.2
#                 correction_right = 0.2
#                 amp_factor = 1.0
#                 for batch_sample in batch_samples:
#                     name = 'data/' + batch_sample[0].split('\')[-3] + '/IMG/' + batch_sample[0].split('\')[-1]
#                     center_image = cv2.imread(name)
#                     center_angle = float(batch_sample[3])
#                     images.append(center_image)
#                     angles.append(center_angle)
#                     name = 'data/' + batch_sample[1].split('\')[-3] + '/IMG/' + batch_sample[1].split('\')[-1]
#                     left_image = cv2.imread(name)
#                     left_angle = float(batch_sample[3]) + correction_left
#                     images.append(left_image)
#                     angles.append(left_angle)
#                     name = 'data/' + batch_sample[2].split('\')[-3] + '/IMG/' + batch_sample[2].split('\')[-1]
#                     right_image = cv2.imread(name)
#                     right_angle = float(batch_sample[3]) - correction_right
#                     images.append(right_image)
#                     angles.append(right_angle)

#                 for batch_sample_flipped in batch_samples_flipped:
#                     name = 'data/' + batch_sample_flipped[0].split('\')[-3]  + '/IMG/' + batch_sample_flipped[0].split('\')[-1]
#                     center_image_flipped = np.fliplr(cv2.imread(name))
#                     center_angle_flipped = - (float(batch_sample_flipped[3]))
#                     images.append(center_image_flipped)
#                     angles.append(center_angle_flipped)
#                     name = 'data/' + batch_sample_flipped[1].split('\')[-3]  + '/IMG/' + batch_sample_flipped[1].split('\')[-1]
#                     left_image_flipped = np.fliplr(cv2.imread(name))
#                     left_angle_flipped = - (float(batch_sample_flipped[3]) + correction_left)
#                     images.append(left_image_flipped)
#                     angles.append(left_angle_flipped)
#                     name = 'data/' + batch_sample_flipped[2].split('\')[-3]  + '/IMG/' + batch_sample_flipped[2].split('\')[-1]
#                     right_image_flipped = np.fliplr(cv2.imread(name))
#                     right_angle_flipped = - (float(batch_sample_flipped[3]) - correction_right)
#                     images.append(right_image_flipped)
#                     angles.append(right_angle_flipped)

#                 # trim image to only see section with road
#                 X_train = np.array(images)
#                 y_train = np.array(angles)*amp_factor
#                 yield sklearn.utils.shuffle(X_train, y_train)

#     # this produces 6 times the size of the orignal data
#     aug_factor = 6
#     print('Augmented Sample Size = {}'.format(len(samples) * aug_factor))

#     # Train and Validation data Generator
#     train_generator = generator(train_samples, batch_size=32)
#     validation_generator = generator(validation_samples, batch_size=32)


#     # Deinfe Convolutional Neural Network in Keras
#     # Nvidia version, proven to be effecive
#     model = Sequential()
#     model.add(Lambda(lambda x:x / 255.0 - 0.5,input_shape=(160,320,3)))
#     model.add(Cropping2D(cropping=((50,25), (0,0))))
#     model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation='relu'))
#     model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation='relu'))
#     model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation='relu'))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1))

#     # Here is some code for fine-tuning and layer freezing

#     # model = Sequential()
#     # del model
#     # model = load_model('model.h5')

#     # Freeze some layers for fine tuning

#     # for layer in model.layers:
#     #     layer.trainable = False
#     # model.layers[-4].trainable = True
#     # model.layers[-3].trainable = True
#     # model.layers[-2].trainable = True
#     # model.layers[-1].trainable = True

#     # Train Model
#     adam = keras.optimizers.Adam(lr=args.lr_late, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
#     model.compile(loss = 'mse',optimizer = adam)
#     model.fit_generator(train_generator, samples_per_epoch= aug_factor*len(train_samples), validation_data=validation_generator,\
#                         nb_val_samples=aug_factor*len(validation_samples), nb_epoch=args.epcohs)

#     #Save the Model
#     model.save('model_new.h5')
#     print('model is saved')

# if __name__ == '__main__':
#     main()
