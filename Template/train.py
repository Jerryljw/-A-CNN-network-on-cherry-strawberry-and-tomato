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
from keras import regularizers
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

epoch=100
batch_size =32 #15 for CNN2 32 forCNN 1
img_length_width =256  # 224 for VGG16 256 for CNN
train_data_path = 'D:\QQPCMgr(1)\Desktop\COMP309_2019_Project\Train_data_enriched_cleaned'
img_category = ['cherry', 'strawberry', 'tomato']


def construct_model(model_type ='model type'):
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    # 1. Baseline model:
    if model_type=='Baseline': # 1 Convolutional layer CNN
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(img_length_width, img_length_width, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    # 2. MLP model
    elif model_type=='MLP':
        model = Sequential()
        model.add(Flatten(input_shape=(img_length_width, img_length_width, 3)))
        model.add(Dense(32))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    # 3. CNN model
    elif model_type == 'CNN':
        model = Sequential()
        # Convolutional layers and pooling layers
        # 1st first hidden layer, convolution- input image, applying feature detectors(feature mapping)
        model.add(Conv2D(32, (3, 3), input_shape=(img_length_width, img_length_width, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 2nd hidden layer
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        # 3rd hidden layer
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        # 4th hidden layer
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        # 5th hidden layer
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        # flatten layer
        model.add(Flatten())
        # fully connected layers
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        # output layer
        model.add(Dense(3,activation='softmax'))
        model.compile(optimizer = Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        # 4. VGG16 pretrained model
    elif model_type =='VGG16':
        # VGG16
        vgg16_model=keras.applications.vgg16.VGG16(include_top=False,input_shape=(img_length_width,img_length_width,3), weights='imagenet')
        model=Sequential()
        # add layers
        for layers in vgg16_model.layers:
            model.add(layers)
        # fully connected layers
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(4096,activation='relu'))
        for layer in model.layers:
            layer.trainable=False
        model.add(Dense(3,activation='softmax'))
        optimizer= Adam(lr=0.0001)#RMSprop(lr=0.0001,rho=0.9) #Adam(lr=0.001)
        model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model and the trained model history(list)
    """
    # Add your code here
    print('start reading data')
    #preprocessing image data
    datagen = ImageDataGenerator(
        validation_split=0.2,# split of validation set
        rescale=1./255, # scale the RGB numbers
        # zoom_range=0.2, # below are preprocessing terms
        # shear_range=0.1,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        horizontal_flip=True
        )
    # generate train data flow
    train_generator = datagen.flow_from_directory(
       directory=train_data_path,
        class_mode='categorical',
       #color_mode="grayscale",
       target_size=(img_length_width, img_length_width),
       batch_size=batch_size,
       classes=img_category,
       subset='training')
    # generate validation data flow
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255)
    validation_generator = datagen.flow_from_directory(
       directory=train_data_path,
        class_mode='categorical',
       #color_mode="grayscale",
       target_size=(img_length_width, img_length_width),
       batch_size=batch_size,
       classes=img_category,
        subset='validation')
    print('start training model')

    # callbacks attributes, some function terms
    modelcheckpoint = callbacks.ModelCheckpoint(filepath="model/model.h5",monitor='val_loss')
    remoteMonitor = callbacks.RemoteMonitor()
    progbarlogger = callbacks.ProgbarLogger(count_mode='steps')
    csvlogger = callbacks.CSVLogger('training{}.log'.format(int(time.time())))
    earlystopping = callbacks.EarlyStopping(monitor='val_loss',verbose=0,restore_best_weights=True)
    learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                          minlr=0.00001)
    NAME='CNN_3{}.log'.format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/tensorboard{}'.format(NAME))
    # fit the model
    model_history = model.fit_generator(train_generator,
                                         steps_per_epoch=len(train_generator),
                                         validation_data=validation_generator,
                                        validation_steps=len(validation_generator),
                                         epochs=epoch,
                                         verbose=2,
                                         callbacks=[tensorboard],
                                        )
    return model,model_history


def save_model(model,model_type):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    model_name = 'model/'+ model_type+ '_'+str(int(time.time())) +'model.h5'
    model.save(model_name)
    print("Model Saved Successfully.")


if __name__ == '__main__':
    #record time
    timing_start = datetime.datetime.now()
    timing_tag_start = str(timing_start.day) + '_' + str(timing_start.hour) + '_' + str(timing_start.minute)
    print('start training at '+timing_tag_start)
    model_type = 'CNN' #'Baseline', 'CNN', ‘VGG16’, 'MLP

    # construct model
    model = construct_model(model_type)
    model,model_history = train_model(model)
    save_model(model,model_type)
    print(model.summary())
    #end time
    timing_end = datetime.datetime.now()
    timing_tag_end = str(timing_end.day) + '_' + str(timing_end.hour) + '_' + str(timing_end.minute)
    print('The Training Time:'+str(timing_end - timing_start));
    print('start drawing graphs')

    # #save accuracy curve
    # plt.plot(model_history.history['acc'])
    # plt.plot(model_history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['train','test'],loc='upper left')
    # plt.savefig('Visualization/Accuracy_{}.png'.format(model_type+str(int(time.time()))))
    # #save loss curve
    # plt.clf()
    # plt.plot(model_history.history['loss'])
    # plt.plot(model_history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train','test'],loc='upper left')
    # plt.savefig('Visualization/loss_{}.png'.format(model_type+str(int(time.time()))))