
# coding: utf-8

# In[22]:


import os
import csv
import cv2
import numpy as np
import sklearn

from random import shuffle

from sklearn.model_selection import train_test_split


# In[38]:



samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_samples = train_samples[0:500]
validation_samples = validation_samples[0:500]


# In[51]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(1, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/IMG/'+batch_samples[0].split('/')[-1]
                center_image = cv2.imread(name)
                image_RGB = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(image_RGB)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            print(len(X_train))
            yield X_train, y_train


# In[52]:


# compile and train the model using the generator function
train_generator = generator(train_samples[:500], batch_size=32)
validation_generator = generator(validation_samples[:500], batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3),output_shape=(160,320,3)))
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, input_shape=(160,320,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)

model.save('model1.h5')


# In[14]:


from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.core import Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[30]:


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, input_shape=(160,320,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
          


# In[31]:


model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2, shuffle=True,nb_epoch=1)
model.save('model.h5')

