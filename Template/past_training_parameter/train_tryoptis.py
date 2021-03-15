#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Created by Yiming Peng and Bing Xue
"""
import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'#only for version error in my own PC
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

import numpy as np
import tensorflow as tf
import random
import cv2
import os
from imutils import paths
from contextlib import redirect_stdout
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input,Flatten,Dropout,Conv2D,MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation
from keras.optimizers import Adam,RMSprop,SGD
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras import callbacks
from keras.callbacks import TensorBoard
from keras.regularizers import l1
import pickle
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time
import matplotlib.pyplot as plt
import keras
import datetime
# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

epoch=30
batch_size =15
img_length_width =256 # 224 for VGG16 150 for CNN
train_data_path = 'D:\QQPCMgr(1)\Desktop\COMP309_2019_Project\Train_data_enriched'
img_category = ['cherry', 'strawberry', 'tomato']

def differentopti(name = 'name'):
    model = Sequential()
    # Convolutional layers and pooling layers
    # 1st convolution- input image, applying feature detectors(feature mapping)
    model.add(Conv2D(32, (3, 3), input_shape=(img_length_width, img_length_width, 3), activation='relu'))
    # 1st pooling- feature map, pooled feature map, reduce complexity and size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2nd convolution layer $ pooling layer
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 3rd convolution layer $ pooling layer
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dense(32))
    # model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))
    if name == 'adam':
        optimizer = Adam(lr=0.0001)
    elif name == 'SGD':
        optimizer=SGD(lr=0.0001)
    elif name =='RMS':
        optimizer = RMSprop(lr=0.0001,rho=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here
    print('start reading data')
    #preprocessing image data
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255)
    train_generator = datagen.flow_from_directory(
       directory=train_data_path,
        class_mode='categorical',
       #color_mode="grayscale",
       target_size=(img_length_width, img_length_width),
       batch_size=batch_size,
       classes=img_category,
       subset='training')

    validation_generator = datagen.flow_from_directory(
       directory=train_data_path,
        class_mode='categorical',
       #color_mode="grayscale",
       target_size=(img_length_width, img_length_width),
       batch_size=batch_size,
       classes=img_category,
        subset='validation')
    print('start training model')
    # callbacks attributes
    model_history = model.fit_generator(train_generator,
                                        steps_per_epoch=len(train_generator),
                                        validation_data=validation_generator,
                                       validation_steps=len(validation_generator),
                                        epochs=epoch,
                                        verbose=2,
                                        )
    # model_history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epoch, verbose=1)
    return model,model_history


if __name__ == '__main__':
    #record time
    timing_start = datetime.datetime.now()
    timing_tag_start = str(timing_start.day) + '_' + str(timing_start.hour) + '_' + str(timing_start.minute)
    print('start training at '+timing_tag_start)
    model1 = differentopti('RMS')
    model2 = differentopti('adam')
    model3 = differentopti('SGD')
    model1,model_history1 = train_model(model1)
    model2, model_history2 = train_model(model2)
    model3, model_history3 = train_model(model3)
    #end time
    timing_end = datetime.datetime.now()
    timing_tag_end = str(timing_end.day) + '_' + str(timing_end.hour) + '_' + str(timing_end.minute)
    print('The Training Time:'+str(timing_end - timing_start));
    print('start drawing graphs')

    #plot accuracy curve
    plt.plot(model_history1.history['acc'])
    plt.plot(model_history1.history['val_acc'])
    plt.plot(model_history2.history['acc'])
    plt.plot(model_history2.history['val_acc'])
    plt.plot(model_history3.history['acc'])
    plt.plot(model_history3.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['RMStrain','RMStest','adamtrain','adamtest','SGDtrain','SGDtest'],loc='upper left')
    plt.savefig('D:\QQPCMgr(1)\Desktop\COMP309_2019_Project\Template\Visualization\optAccuracy_{}.png'.format(int(time.time())))
    #plot loss curve
    plt.clf()
    plt.plot(model_history1.history['loss'])
    plt.plot(model_history1.history['val_loss'])
    plt.plot(model_history2.history['loss'])
    plt.plot(model_history2.history['val_loss'])
    plt.plot(model_history3.history['loss'])
    plt.plot(model_history3.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['RMStrain','RMStest','adamtrain','adamtest','SGDtrain','SGDtest'],loc='upper left')
    plt.savefig('D:\QQPCMgr(1)\Desktop\COMP309_2019_Project\Template\Visualization\optloss_{}.png'.format(int(time.time())))