#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import re
import shutil
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
import numpy as np


# In[2]:

clean_test_lab=pd.read_csv('clean_test_label.csv')
clean_train_lab=pd.read_csv('clean_train_label.csv')
# In[4]:


from keras.preprocessing import image
from vergeml.img import resize_image

target_size = (224, 224)
def getImagePixels(file):
    img = image.load_img(file, grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x
 
clean_train_lab['pixels'] = clean_train_lab['file'].progress_apply(getImagePixels)
clean_test_lab['pixels'] = clean_test_lab['file'].progress_apply(getImagePixels)

train_features = []
 
for i in range(0, clean_train_lab.shape[0]):
    train_features.append(clean_train_lab['pixels'].values[i])

train_features = np.array(train_features)
train_features = train_features.reshape(train_features.shape[0], 224, 224, 3)
 
train_features = train_features / 255

test_features = []

for i in range(0, clean_test_lab.shape[0]):
    test_features.append(clean_test_lab['pixels'].values[i])

test_features = np.array(test_features)
test_features = test_features.reshape(test_features.shape[0], 224, 224, 3)
test_features = test_features / 255

train_label = clean_train_lab[['race']]
test_label = clean_test_lab[['race']]
 
races = clean_train_lab['race'].unique()
 
for j in range(len(races)): #label encoding
    current_race = races[j]
    print("replacing ",current_race," to ", j+1)
    train_label['race'] = train_label['race'].replace(current_race, str(j+1))
    test_label['race'] = test_label['race'].replace(current_race, str(j+1))
train_label = train_label.astype({'race': 'int32'})
test_label = test_label.astype({'race': 'int32'})
 
train_target = pd.get_dummies(train_label['race'], prefix='race')
test_target = pd.get_dummies(test_label['race'], prefix='race')

#Dataset spliting
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(
    train_features, train_target.values
    , test_size=0.12, random_state=17
)


from keras import layers,Sequential
from keras.layers.convolutional import ZeroPadding2D,Convolution2D,MaxPooling2D
from keras.layers import Dropout,Flatten,Activation
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
 

model.load_weights('vgg_face_weights.h5')

#Transfer Learning
from keras import Model
for layer in model.layers[:-7]:
    layer.trainable = False
num_of_classes=6
base_model_output = Sequential()
base_model_output = Convolution2D(num_of_classes, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)
 
race_model = Model(inputs=model.input, outputs=base_model_output)


import keras
from keras.callbacks import ModelCheckpoint

epochs=40
race_model.compile(loss='categorical_crossentropy'
, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
 
checkpointer = ModelCheckpoint(filepath='race_model_single_batch.hdf5'
, monitor = "val_loss", verbose=1, save_best_only=True, mode = 'auto')


for i in range(0, epochs):
    print("Epoch ", i, ". ", end='')
    ix_train = np.random.choice(train_x.shape[0], size=batch_size)
 
    score = race_model.fit(
        train_x[ix_train], train_y[ix_train]
        , epochs=1
        , validation_data=(val_x, val_y)
        , callbacks=[checkpointer]
    )
 
    val_loss = score.history['val_loss'][0]; train_loss = score.history['loss'][0]
    val_scores.append(val_loss); train_scores.append(train_loss)
 
    if val_loss < loss:
        loss = val_loss * 1
        last_improvement = 0
        best_iteration = i * 1
    else:
        last_improvement = last_improvement + 1
        print("try to decrease val loss for ",patience - last_improvement," epochs more")
 
    if last_improvement == patience:
        print("there is no loss decrease in validation for ",patience," epochs. early stopped")
        break

 
batch_size = pow(2, 14); patience = 50
last_improvement = 0; best_iteration = 0
loss = 1000000 #initialize as a large value
 
