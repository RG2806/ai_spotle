#!/usr/bin/env python


import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/source/')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.models import Sequential #Initialise our neural network model as a sequential network
from keras.regularizers import l2
from keras.layers import Activation#Applies activation function
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score

import pandas as pd
import keras 
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras.backend as k
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import os
import keras




'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''

def  aithon_level2_api(trainingcsv, testcsv):
	classes = ['Fear','Sad','Happy']
  	data = []
  	labels =[]
  	df=pd.read_csv(trainingcsv)
  	for i,row in df.iterrows():
      		image_data=np.asarray([int(x) for x in row[1:]]).reshape(48,48)
      		image_data =image_data.astype(np.float32)/255.0
      		data.append(image_data)
      		labels.append(classes.index(row[0]))
      		data.append(cv2.flip(image_data, 1))
      		labels.append(classes.index(row[0]))
  	data = np.expand_dims(data, -1)   
  	labels = to_categorical(labels, num_classes = 3)
	train_data=np.array(data)
	train_labels=np.array(labels)
	x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=101,shuffle=True)
	num_features = 64
	width, height = 48, 48
	model = Sequential()

	model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
	model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(2*2*2*num_features, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(2*2*num_features, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(2*num_features, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),metrics=['accuracy'])

	lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
	checkpointer = ModelCheckpoint('model.h5', monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
	callbacks = [lr_reducer, checkpointer]
	bs = 64
	epochs = 100

	aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, fill_mode="nearest")

	H = model.fit(aug.flow(x_train, y_train, batch_size=bs), validation_data=(x_test, y_test), steps_per_epoch=len(x_train)//bs, callbacks=callbacks, shuffle=True, epochs=epochs)
	model.load_weights('model.h5')
	test_data = []
  	df1=pd.read_csv(testcsv)
	if 'emotion' in df1.columns:
    		df1=df1.drop(['emotion'], axis = 1) 
  	for i,row in df1.iterrows():
      		image_data=np.asarray([int(x) for x in row[0:]]).reshape(48,48)
      		image_data =image_data.astype(np.float32)/255.0
      		test_data.append(image_data)
  	test_data = np.expand_dims(test_data, -1)   
	test_data=np.array(test_data)
	y_pred=model.predict(test_data)
	y_pred=y_pred.argmax(axis=1)
	result=[]
	for i in y_pred:
		result.append(classes[i])
	return result










