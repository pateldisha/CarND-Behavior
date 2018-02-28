
# coding: utf-8

# In[1]:


#Importing the libraries
import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

from random import shuffle

from sklearn.model_selection import train_test_split


# In[2]:


#Import from Keras library

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda, Dropout, Activation
from keras.layers.core import Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


# In[3]:


# Function to reduce the brightness 

def brightness_images(image):

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# #### Reading the pre-processing images

# In[ ]:


images = []
measurements = []
lines = []

# Fetching the image details from the log CSV file
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Reading all the images 

for line in lines:
    # Reading and processing Center Images
    image = 'data/IMG/'+line[0].split('/')[-1]
    image = mpimg.imread(image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
	# Calling the function for applying brightness reduction
    image = brightness_images(image)
    images.append(image)
    measurements.append(measurement)
    
    # Reading and processing Left images
    image = 'data/IMG/'+line[1].split('/')[-1]
    image = mpimg.imread(image)
    images.append(image)
    # Applying correction factor of 0.25 to the angle for the left images
    measurement = float(line[3]) + 0.25
    measurements.append(measurement)
    image = brightness_images(image)
    images.append(image)
    measurements.append(measurement)
    
    # Reading and processing Right Images
    image = 'data/IMG/'+line[2].split('/')[-1]
    image = mpimg.imread(image)
    images.append(image)
    # Applying correction factor of 0.25 to the angle for the right images
    measurement = float(line[3]) - 0.25
    measurements.append(measurement)
    image = brightness_images(image)
    images.append(image)
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Training the Model

# In[7]:


# Train the model 

ch, row, col = 3, 160, 320

model = Sequential()
#Applying normalization 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
#Cropping the image - 70 pixels from the top and 25 pixels from the bottom
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row,col,ch)))
# Convolution network with filter 24 and RELU activation
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
# Convolution network with filter 36 and RELU activation
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
# Convolution network with filter 48 and RELU activation
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
# Convolution network with filter 64 and RELU activation
model.add(Convolution2D(64,3,3, activation='relu'))
# Convolution network with filter 64 and RELU activation
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
# Fully connected network with 1000 neurons
model.add(Dense(1000))
# Applying dropout 
model.add(Dropout(0.5))
# RELU activation
model.add(Activation('relu'))
# Fully connected network with 500 neurons
model.add(Dense(500))
# Fully connected network with 100 neurons
model.add(Dense(100))
# Fully connected network with 50 neurons
model.add(Dense(50))
# Fully connected network with 20 neurons
model.add(Dense(20))
# Fully connected network with 10 neurons
model.add(Dense(10))
# Fully connected network with 1 neuron
model.add(Dense(1))

#Compile the model
model.compile(loss = 'mse', optimizer= 'adam')
# Training the model
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

#Saving the model
model.save('model.h5')

#Fetch the model summary
model.summary()

