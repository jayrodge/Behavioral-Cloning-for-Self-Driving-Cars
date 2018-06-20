import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Activation
from keras.layers import Lambda
from keras import optimizers
from keras.layers.convolutional import MaxPooling2D,Convolution2D,Cropping2D,Conv2D

lines=[]
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
lines=lines[1:]
images=[]
measurements=[]
for line in lines:
	for i in range(3):
		source_path=line[i]
		filename=source_path.split('/')[-1]
		current_path='data/IMG/'+filename
		image = cv2.imread(current_path)
		images.append(image)
	measurement = float(line[3])
	correction=0.2
	measurements.append(measurement)
	measurements.append(measurement+correction)
	measurements.append(measurement-correction)

image_flipped=[]
measurement_flipped=[]
for i in range(len(images)):
    image_flipped.append(np.fliplr(images[i]))
    measurement_flipped.append(-measurements[i])

images=images+image_flipped
measurements=measurements+measurement_flipped

del image_flipped
del measurement_flipped

X_train=np.array(images)
y_train=np.array(measurements)
print("Array Done")
model=Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(6, 5, 5,activation='relu',))
print('Convolute')
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))
print('Model Done')
model.compile(loss='mse',optimizer='adam')
model.summary()
model.fit(X_train,y_train,nb_epoch=3,validation_split=0.2,shuffle=True)
model.save('models/model_final.h5')
exit()	