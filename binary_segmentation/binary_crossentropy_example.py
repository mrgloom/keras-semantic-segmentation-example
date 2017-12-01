# coding: utf-8

import os
import sys
import math
import random as rn

import cv2
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.optimizers import Adadelta, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

import models

#Parameters
INPUT_CHANNELS = 3
NUMBER_OF_CLASSES = 1
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
            cv2.ellipse(img, (int(center_x), int(center_y)), (int(r_x), int(r_y)), int(0), int(0), int(360), (int(obj1_color0), int(obj1_color1), int(obj1_color2)), int(-1))
            cv2.ellipse(mask, (int(center_x), int(center_y)), (int(r_x), int(r_y)), int(0), int(0), int(360), int(255), int(-1))
            break
    
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
    cv2.imshow('mask object 1',y_pred[:,:,0])
    cv2.waitKey(0)

def save_prediction():
    
    model = get_model()
    model.load_weights('model_weights_'+loss_name+'.h5')
    
    img,mask= gen_random_image()
    
    y_pred= model.predict(img[None,...].astype(np.float32))[0]
    
    print('y_pred.shape', y_pred.shape)
    
    y_pred= y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))

    print('np.min(mask[:,:,0])', np.min(mask[:,:,0]))
    print('np.max(mask[:,:,0])', np.max(mask[:,:,0]))
        
    print('np.min(y_pred)', np.min(y_pred))
    print('np.max(y_pred)', np.max(y_pred))
    
    res = np.zeros((IMAGE_H,4*IMAGE_W,3),np.uint8)
    res[:,:IMAGE_W,:] = img
    res[:,IMAGE_W:2*IMAGE_W,:] = cv2.cvtColor(mask[:,:,0],cv2.COLOR_GRAY2RGB)
    res[:,2*IMAGE_W:3*IMAGE_W,:] = 255*cv2.cvtColor(y_pred[:,:,0],cv2.COLOR_GRAY2RGB)
    y_pred[:,:,0][y_pred[:,:,0] > 0.5] = 255
    res[:,3*IMAGE_W:4*IMAGE_W,:] = cv2.cvtColor(y_pred[:,:,0],cv2.COLOR_GRAY2RGB)
    
    cv2.imwrite(loss_name+'_result.png', res)
    
def visualy_inspect_generated_data():
    img,mask = gen_random_image()
    
    cv2.imshow('img',img)
    cv2.imshow('mask object 1',mask[:,:,0])
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
        samples_per_epoch=100,
        validation_data=batch_generator(batch_size),
        nb_val_samples=10,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)
        
if __name__ == '__main__':
    #visualy_inspect_generated_data()
    
    train()
    #visualy_inspect_result()
    save_prediction() 
    
    
    
    
    
