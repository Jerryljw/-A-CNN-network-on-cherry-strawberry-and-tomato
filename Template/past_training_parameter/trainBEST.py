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
img_length_width =256 # 224 for VGG16 256 for CNN
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
    if model_type=='Baseline':
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
    # 2. CNN model
    elif model_type == 'CNN1':
        model = Sequential()
        # Convolutional layers and pooling layers
        # 1st convolution- input image, applying feature detectors(feature mapping)
        model.add(Conv2D(32, (3, 3), input_shape=(img_length_width, img_length_width, 3), activation='relu'))
        #1st pooling- feature map, pooled feature map, reduce complexity and size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 2nd convolution layer $ pooling layer
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
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
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(3,activation='softmax'))
        model.compile(optimizer = Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
    elif model_type == 'CNN2':
        model = Sequential()
        # Convolutional layers and pooling layers
        # 1st convolution- input image, applying feature detectors(feature mapping)
        model.add(Conv2D(32, (3, 3), input_shape=(img_length_width, img_length_width, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        # 2nd convolution layer $ pooling layer
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
    elif model_type =='VGG16':
        # VGG16
        vgg16_model=keras.applications.vgg16.VGG16(include_top=False,input_shape=(img_length_width,img_length_width,3), weights='imagenet')
        model=Sequential()
        for layers in vgg16_model.layers:
            model.add(layers)
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(4096,activation='relu'))
        for layer in model.layers:
            layer.trainable=False # probably don't need this in here
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
    :return:model:   the trained CNN model
    """
    # Add your code here
    print('start reading data')
    #preprocessing image data
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        # zoom_range=0.2,
        # shear_range=0.1,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        horizontal_flip=True
        )
    train_generator = datagen.flow_from_directory(
       directory=train_data_path,
        class_mode='categorical',
       #color_mode="grayscale",
       target_size=(img_length_width, img_length_width),
       batch_size=batch_size,
       classes=img_category,
       subset='training')
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
    # callbacks attributes
    modelcheckpoint = callbacks.ModelCheckpoint(filepath="model/model.h5",monitor='val_loss')
    remoteMonitor = callbacks.RemoteMonitor()
    progbarlogger = callbacks.ProgbarLogger(count_mode='steps')
    csvlogger = callbacks.CSVLogger('training{}.log'.format(int(time.time())))
    earlystopping = callbacks.EarlyStopping(monitor='val_loss',verbose=0,restore_best_weights=True)
    learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                          minlr=0.00001)

    NAME='CNN_3{}.log'.format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    model_history = model.fit_generator(train_generator,
                                         steps_per_epoch=len(train_generator),
                                         validation_data=validation_generator,
                                        validation_steps=len(validation_generator),
                                         epochs=epoch,
                                         verbose=2,
                                         #callbacks=[earlystopping],
                                        )
    #model_history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=epoch, verbose=1)
    return model,model_history


def save_model(model,model_type):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model_name = 'model/'+str(int(time.time()))+'_'+ model_type +'model.h5'
    model.save(model_name)
    print("Model Saved Successfully.")


if __name__ == '__main__':
    #record time
    timing_start = datetime.datetime.now()
    timing_tag_start = str(timing_start.day) + '_' + str(timing_start.hour) + '_' + str(timing_start.minute)
    print('start training at '+timing_tag_start)
    model_type = 'CNN1' #'Baseline', 'CNN1', ‘VGG16’，CNN2
    model = construct_model(model_type)
    model,model_history = train_model(model)
    save_model(model,model_type)
    print(model.summary())
    #end time
    timing_end = datetime.datetime.now()
    timing_tag_end = str(timing_end.day) + '_' + str(timing_end.hour) + '_' + str(timing_end.minute)
    print('The Training Time:'+str(timing_end - timing_start));
    print('start drawing graphs')

    #plot accuracy curve
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.savefig('Visualization/Accuracy_{}.png'.format(model_type+str(int(time.time()))))
    #plot loss curve
    plt.clf()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.savefig('Visualization/loss_{}.png'.format(model_type+str(int(time.time()))))