# coding: utf-8

########################################################################################################################################
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
########################################################################################################################################

import sys
import math

import cv2
import pandas as pd

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.optimizers import Adadelta, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

import models

#Parameters
INPUT_CHANNELS = 3
NUMBER_OF_CLASSES = 2
IMAGE_W = 224
IMAGE_H = 224

epochs = 100*1000
patience = 60
batch_size = 8

loss_name = "binary_crossentropy"

def get_model():
    
    inputs = Input((IMAGE_H, IMAGE_W, INPUT_CHANNELS))
    
    base = models.get_fcn_vgg16_32s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_fcn_vgg16_16s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_fcn_vgg16_8s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_unet(inputs, NUMBER_OF_CLASSES)
    #base = models.get_segnet_vgg16(inputs, NUMBER_OF_CLASSES)
    
    # sigmoid
    reshape= Reshape((-1,NUMBER_OF_CLASSES))(base)
    act = Activation('sigmoid')(reshape)
    
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy')
    
    #print(model.summary())
    #sys.exit()
    
    return model
    
def gen_random_image():
    img = np.zeros((IMAGE_H, IMAGE_W, INPUT_CHANNELS), dtype=np.uint8)
    mask = np.zeros((IMAGE_H, IMAGE_W, NUMBER_OF_CLASSES), dtype=np.uint8)
    mask_obj1 = np.zeros((IMAGE_H, IMAGE_W, 1), dtype=np.uint8)
    mask_obj2 = np.zeros((IMAGE_H, IMAGE_W, 1), dtype=np.uint8)
    
    colors = np.random.permutation(256)
    
    # Background
    img[:, :, 0] = colors[0]
    img[:, :, 1] = colors[1]
    img[:, :, 2] = colors[2]

    # Object class 1
    obj1_color0 = colors[3]
    obj1_color1 = colors[4]
    obj1_color2 = colors[5]
    while(True):
        center_x = rn.randint(0, IMAGE_W)
        center_y = rn.randint(0, IMAGE_H)
        r_x = rn.randint(10, 50)
        r_y = rn.randint(10, 50)
        if(center_x+r_x < IMAGE_W and center_x-r_x > 0 and center_y+r_y < IMAGE_H and center_y-r_y > 0):
            cv2.ellipse(img, (center_x, center_y), (r_x, r_y), 0, 0, 360, (obj1_color0, obj1_color1, obj1_color2), -1)
            cv2.ellipse(mask_obj1, (center_x, center_y), (r_x, r_y), 0, 0, 360, 255, -1)
            break
    
    # Object class 2
    obj2_color0 = colors[6]
    obj2_color1 = colors[7]
    obj2_color2 = colors[8]
    while(True):
        left = rn.randint(0, IMAGE_W)
        top = rn.randint(0, IMAGE_H)
        dw = rn.randint(int(10*math.pi), int(50*math.pi))
        dh = rn.randint(int(10*math.pi), int(50*math.pi))
        if(left+dw < IMAGE_W and top+dh < IMAGE_H):
            mask_obj2 = np.zeros((IMAGE_H, IMAGE_W, 1), dtype=np.uint8)
            cv2.rectangle(mask_obj2, (left, top), (left+dw, top+dh), 255, -1)
            if(np.sum(cv2.bitwise_and(mask_obj1,mask_obj2)) == 0):
                cv2.rectangle(img, (left, top), (left+dw, top+dh), (obj2_color0, obj2_color1, obj2_color2), -1)
                break
            
    mask[:,:,0] = np.squeeze(mask_obj1)
    mask[:,:,1] = np.squeeze(mask_obj2)
    
    # White noise
    density = rn.uniform(0, 0.1)
    for i in range(IMAGE_H):
        for j in range(IMAGE_W):
            if rn.random() < density:
                img[i, j, 0] = rn.randint(0, 255)
                img[i, j, 1] = rn.randint(0, 255)
                img[i, j, 2] = rn.randint(0, 255)

    return img, mask

def batch_generator(batch_size):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = gen_random_image()
            image_list.append(img)
            mask_list.append(mask)

        image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0 # [0,1]
        
        mask_list= mask_list.reshape(batch_size,IMAGE_H*IMAGE_W,NUMBER_OF_CLASSES)
                
        yield image_list, mask_list

def visualy_inspect_result():
    
    model = get_model()
    model.load_weights('model_weights_'+loss_name+'.h5')
    
    img,mask= gen_random_image()
    
    y_pred= model.predict(img[None,...].astype(np.float32))[0]
    
    print('y_pred.shape', y_pred.shape)
    
    y_pred= y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))
    
    print('np.min(y_pred)', np.min(y_pred))
    print('np.max(y_pred)', np.max(y_pred))
    
    cv2.imshow('img',img)
    cv2.imshow('mask 1',mask[:,:,0])
    cv2.imshow('mask 2',mask[:,:,1])
    cv2.imshow('mask object 1',y_pred[:,:,0])
    cv2.imshow('mask object 2',y_pred[:,:,1])
    cv2.waitKey(0)

def save_prediction():
    
    model = get_model()
    model.load_weights('model_weights_'+loss_name+'.h5')
    
    img,mask= gen_random_image()
    
    y_pred= model.predict(img[None,...].astype(np.float32))[0]
    
    print('y_pred.shape', y_pred.shape)
    
    y_pred= y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))

    print('np.min(mask[:,:,0])', np.min(mask[:,:,0]))
    print('np.max(mask[:,:,1])', np.max(mask[:,:,1]))
        
    print('np.min(y_pred)', np.min(y_pred))
    print('np.max(y_pred)', np.max(y_pred))
    
    res = np.zeros((IMAGE_H,7*IMAGE_W,3),np.uint8)
    res[:,:IMAGE_W,:] = img
    res[:,IMAGE_W:2*IMAGE_W,:] = cv2.cvtColor(mask[:,:,0],cv2.COLOR_GRAY2RGB)
    res[:,2*IMAGE_W:3*IMAGE_W,:] = cv2.cvtColor(mask[:,:,1],cv2.COLOR_GRAY2RGB)
    res[:,3*IMAGE_W:4*IMAGE_W,:] = 255*cv2.cvtColor(y_pred[:,:,0],cv2.COLOR_GRAY2RGB)
    res[:,4*IMAGE_W:5*IMAGE_W,:] = 255*cv2.cvtColor(y_pred[:,:,1],cv2.COLOR_GRAY2RGB)
    y_pred[:,:,0][y_pred[:,:,0] > 0.5] = 255
    y_pred[:,:,1][y_pred[:,:,1] > 0.5] = 255
    res[:,5*IMAGE_W:6*IMAGE_W,:] = cv2.cvtColor(y_pred[:,:,0],cv2.COLOR_GRAY2RGB)
    res[:,6*IMAGE_W:7*IMAGE_W,:] = cv2.cvtColor(y_pred[:,:,1],cv2.COLOR_GRAY2RGB)
    
    cv2.imwrite(loss_name+'_result.png', res)
    
def visualy_inspect_generated_data():
    img,mask = gen_random_image()
    
    cv2.imshow('img',img)
    cv2.imshow('mask object 1',mask[:,:,0])
    cv2.imshow('mask object 2',mask[:,:,1])
    cv2.waitKey(0)
    
def train():
    model = get_model()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('model_weights_'+loss_name+'.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(
        generator=batch_generator(batch_size),
        nb_epoch=epochs,
        samples_per_epoch=batch_size,
        validation_data=batch_generator(batch_size),
        nb_val_samples=batch_size,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

if __name__ == '__main__':
    #visualy_inspect_generated_data()
    
    train()
    #visualy_inspect_result()
    save_prediction()    
    
    
    
    
    
