from __future__ import print_function

import numpy as np
import keras
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
import prepare_data as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def main(train_file, validation_file):

	#we will test three different learning rates. 
	lr=[0.00001,0.000001, 0.0000001]

	#load data (take a look on how it works)
	#In this case, we will consider 500 32x32 blocks located at the image's top-left
	#there are possible 10 classes in our problem
	#it returns x and y, where
	#x is a matrix containing all 32x32x3 blocks
	#y is a matrix containing the class in a binary form (one-hot matrices) necessary to make keras work
	#groundtruth: the same as y, but it is not in a binary form, we use it to report metrics later

	x_train, y_train, groundtruth_train=load_data(train_file, 500, (32,32), 10)
	x_validation, y_validation, groundtruth_validation=load_data(validation_file, 500, (32,32), 10)

	#for each learning rate, we will use 7 different learning algorithms	
	for rate in lr:

		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'rmsprop')
		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'adam')
		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'adamax')
		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'adadelta')
		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'adagrad')
		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'sgd')
		train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, rate, 'nadam')

def load_data(filename, blocks_per_image, block_size, num_classes):

	#reading data. We will split the top-left part of the image en 28x28 blocks in grayscale.
	#we will extract 500 blocks in this regions
	#it is just a toy example! I am not expecting good results with standard parameters
	# STUDY the prepare_data.py file!
	x, y = pd.prepare_data(filename, blocks_per_image, block_size)
	
	#separating the ground truth to calculate metrics later
	groundtruth=y

	#pre-processing labels (classes)
	y = np_utils.to_categorical(y, num_classes)

	#pre-processing (normalizing) blocks
	x= x.astype('float32')
	x /= 255.

	return x,y,groundtruth

def train_network(x_train, y_train, x_validation, y_validation, groundtruth_validation, learning_rate, optimizer_name):

	#Information necessary to make the network running
	batch_size = 32
	num_classes = 10
	nb_epoch =1
	
	#this is our model (very small network, BTW)
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	convout1=Activation('sigmoid')
        model.add(convout1)
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	
	#lets decide which learning algorithm we will use (try all)
	if (optimizer_name=='rmsprop'):
		optimizer_params= keras.optimizers.rmsprop(lr=learning_rate, decay=0.0005) 
	if (optimizer_name=='adam'):
		optimizer_params=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
	if (optimizer_name=='adamax'):
		optimizer_params=keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	if (optimizer_name=='adadelta'):
		optimizer_params=keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
	if (optimizer_name=='adagrad'):
		optimizer_params=keras.optimizers.Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0)     
	if (optimizer_name=='sgd'):
		optimizer_params = keras.optimizers.SGD(lr=learning_rate, decay=0.0005)   
	if (optimizer_name=='nadam'):
		optimizer_params=keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

	#ok, lets save this model by compiling it
	model.compile(loss='categorical_crossentropy', optimizer=optimizer_params, metrics=["accuracy"])

	#Lets train and validate using the data we have (no data augmentation) 
	model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(x_validation, y_validation))
	predict_validation=model.predict(x_validation, batch_size=32, verbose=0)
	target_names = ['HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'LG-Nexus-5x', 'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X', 'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']
	predict_validation=np.argmax(predict_validation, axis=1)
	report=classification_report(groundtruth_validation, predict_validation, target_names=target_names)
	accuracy=accuracy_score(groundtruth_validation,predict_validation)	
	
	print(report)
	text_file = open("accuracies.txt", "a")
	text_file.write("%f \n" % accuracy)
	text_file.close()

	print('Final Accuracy for this experiment: ' + str(accuracy))
	
