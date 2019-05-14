#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:20:41 2019

@author: kostas
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#build cnn
classifier=Sequential()
#add the convolutional level
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#the step of flatten
classifier.add(Flatten())

#at the end we need to make our ann which is going to predict weather is dog or cat
#full connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the convolutional neural network ot the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/home/kostas/Desktop/Udemy/excersice 2/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/kostas/Desktop/Udemy/excersice 2/dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=2000)
#Single prediction for testing
import numpy as np
from keras.preprocessing import image

test_img=image.load_img('/dataset/single_prediction/cat_or_dog_1.jpg')
test_img=image.img_to_array(test_img)
#add one more dimension for making thing right for predict method
test_img=np.expand_dims(test_img,axis=0)
prediction=classifier.predict(test_img)
#training_set.indices

