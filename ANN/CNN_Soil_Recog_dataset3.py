# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 19:40:16 2023

@author: ludej
"""

training_data_directory = "train"
test_data_directory = 'test'

import os
import re
import cv2 
import time
import shutil
#import zipfile
#import urllib.request
import numpy as np
from PIL import Image #PIL = Python Image Library
from os import listdir
from os.path import isfile, join
from random import randrange 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 


#Initiate data processing tools
training_data_processor = ImageDataGenerator(
    rescale = 1./255,#This rescales all our values so that they range from 0 to 1
    horizontal_flip = True,
    zoom_range = 0.2,
    rotation_range = 10,
    shear_range = 0.2,
    height_shift_range = 0.1,
    width_shift_range = 0.1
    
    
    )

test_data_processor = ImageDataGenerator(rescale= 1./255)


#Load data into Python
training_data = training_data_processor.flow_from_directory(
    training_data_directory,
    target_size = (256,256),
    batch_size = 32,
    class_mode = 'categorical',
    
    )


testing_data = test_data_processor.flow_from_directory(
    test_data_directory,
    target_size = (256,256),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = False
    )


#Choose model parameters
num_conv_layers = 2
num_dense_layers = 1
layer_size = 32
num_training_epochs = 40 
MODEL_NAME = 'soil'


#Initiate model variable
model = Sequential()

#Begin adding properties to model variable
#e.g. adda convolutional layer
model.add(Conv2D(layer_size, (3,3), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


#add additional convolutional layers based on num_conv_layers
for _ in range(num_conv_layers-1):
    model.add(Conv2D(layer_size, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
#reduce dimensionality
model.add(Flatten())


#add fully connected "dense" Layers if specified
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
    

#add output Layer
model.add(Dense(1)) #Leo - Changed this to 1
model.add(Activation('softmax'))


#Compile the sequential model with all added properties
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              
              )


#use the data already loaded previously to train/tune the model
model.fit(training_data,
          epochs=num_training_epochs,
          validation_data = testing_data
          )


#save the trained model
model.save(f'{MODEL_NAME}.h5')





























