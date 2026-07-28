# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:38:18 2023

@author: Gong Ming
"""

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras.utils import normalize, to_categorical 

from data3 import dice_coef, bce_dice_loss, dice_loss

#keras.utils.conv_utils.normalize_data_format


def conv(inputs,filters):
    conv1 = Conv2D(filters, 3, use_bias=False, padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)
    
    conv2 = Conv2D(filters, 3, use_bias=False, padding = 'same', kernel_initializer='he_normal')(act1)
    conv2 = BatchNormalization()(conv2)
    act2 = Activation("relu")(conv2)

    
    return act2

def fconv(inputs,filters):
    conv1 = Conv2D(filters, 3,strides=(1,1), use_bias=False, padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)

    
    return act1


def dconv(inputs,filters):
    conv1 = Conv2D(filters, 3,strides=(2,2), use_bias=False, padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)

    
    return act1

def dconv1(inputs,filters):
    conv1 = Conv2D(filters, 3,strides=(4,4), use_bias=False, padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)

    
    return act1

def dconv2(inputs,filters):
    conv1 = Conv2D(filters, 3,strides=(8,8), use_bias=False, padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(conv1)

    
    return act1




def resnet(v1,v2,filters):
    cov = Conv2D(filters, 1, padding="same", activation='relu')(v1)
    ext = Add()([cov,v2])
    
    return ext



def unet(pretrained_weights = None,input_size = (64,64,1)):
    inputs = Input(input_size)
    inputs1 = Input(input_size)
    inp = concatenate([inputs,inputs1], axis = 3)
    

    """stage1 """
    
    conv1_1_1 = conv(inp,filters=64)
    res1_1_1 = resnet(inp,conv1_1_1,filters=64)
    conv1_1_2 = conv(res1_1_1,filters=64)
    res1_1_2 = resnet(res1_1_1,conv1_1_2,filters=64)
    
    """stage2 """
    
    conv1_2_1 = conv(res1_1_2,filters=64)
    conv2_2_1 = dconv(res1_1_2,filters=128)
    
    
    conv1_2_2 = conv(conv1_2_1,filters=64)
    res1_2_2 = resnet(conv1_2_1,conv1_2_2,filters=64)
    conv1_2_3 = conv(res1_2_2,filters=64)
    res1_2_3 = resnet(res1_2_2,conv1_2_3,filters=64)
    
    
    conv2_2_2 = conv(conv2_2_1,filters=128)
    res2_2_2 = resnet(conv2_2_1,conv2_2_2,filters=128)
    conv2_2_3 = conv(res2_2_2,filters=128)
    res2_2_3 = resnet(res2_2_2,conv2_2_3,filters=128)
    
    
    """stage3 """
    
    
    conv1_3_1 = conv(res1_2_3,filters=64)
    conv2_3_1 = conv(res2_2_3,filters=128)
    conv3_3_1 = dconv(res2_2_3,filters=256)
    
    d1_res1_2_3 = dconv(res1_2_3,filters=128)
    d2_res1_2_3 = dconv1(res1_2_3,filters=256)
    u1_res2_2_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res2_2_3))
          
    fus1_1 = concatenate([conv1_3_1,u1_res2_2_3], axis = 3)
    fus2_1 = concatenate([conv2_3_1,d1_res1_2_3], axis = 3)
    fus3_1 = concatenate([conv3_3_1,d2_res1_2_3], axis = 3)
    
    fuse1_1 = conv(fus1_1,filters=64)
    fuse2_1 = conv(fus2_1,filters=128)
    fuse3_1 = conv(fus3_1,filters=256)
    
    
    conv1_3_2 = conv(fuse1_1,filters=64)
    res1_3_2 = resnet(fuse1_1,conv1_3_2,filters=64)
    conv1_3_3 = conv(conv1_3_2,filters=64)
    res1_3_3 = resnet(res1_3_2,conv1_3_3,filters=64)
    
    
    conv2_3_2 = conv(fuse2_1,filters=128)
    res2_3_2 = resnet(fuse2_1,conv2_3_2,filters=128)
    conv2_3_3 = conv(res2_3_2,filters=128)
    res2_3_3 = resnet(res2_3_2,conv2_3_3,filters=128)
    
    
    conv3_3_2 = conv(fuse3_1,filters=256)
    res3_3_2 = resnet(fuse3_1,conv3_3_2,filters=256)
    conv3_3_3 = conv(res3_3_2,filters=256)
    res3_3_3 = resnet(res3_3_2,conv3_3_3,filters=256)
    
    
    
    
    """stage4 """
    
    
    conv1_4_1 = conv(res1_3_3,filters=64)
    conv2_4_1 = conv(res2_3_3,filters=128)
    conv3_4_1 = conv(res3_3_3,filters=256)
    conv4_4_1 = dconv(res3_3_3,filters=512)
    
    
    d1_res1_3_3 = dconv(res1_3_3,filters=128)
    d2_res1_3_3 = dconv1(res1_3_3,filters=256)
    d3_res1_3_3 = dconv2(res1_3_3,filters=512)
    
    u1_res2_3_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res2_3_2))
    d1_res2_3_3 = dconv(res2_3_2,filters=256)
    d2_res2_3_3 = dconv1(res2_3_2,filters=512)
    
    
    u1_res3_3_3 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res3_3_2))
    u2_res3_3_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(res3_3_2))
    
    
          
    fus1_2 = concatenate([conv1_4_1,u1_res2_3_3,u2_res3_3_3], axis = 3)
    fus2_2 = concatenate([d1_res1_3_3,conv2_4_1,u1_res3_3_3], axis = 3)
    fus3_2 = concatenate([d2_res1_3_3,d1_res2_3_3,conv3_4_1], axis = 3)
    fus4_2 = concatenate([d3_res1_3_3,d2_res2_3_3,conv4_4_1], axis = 3)
    
    fuse1_2 = conv(fus1_2,filters=64)
    fuse2_2 = conv(fus2_2,filters=128)
    fuse3_2 = conv(fus3_2,filters=256)
    fuse4_2 = conv(fus4_2,filters=512)
    
    conv1_4_2 = conv(fuse1_2,filters=64)
    res1_4_2 = resnet(fuse1_2,conv1_4_2,filters=64)
    conv1_4_3 = conv(res1_4_2,filters=64)
    res1_4_3 = resnet(res1_4_2,conv1_4_3,filters=64)
    
    
    conv2_4_2 = conv(fuse2_2,filters=128)
    res2_4_2 = resnet(fuse2_2,conv2_4_2,filters=128)
    conv2_4_3 = conv(res2_4_2,filters=128)
    res2_4_3 = resnet(res2_4_2,conv2_4_3,filters=128)
    
    
    conv3_4_2 = conv(fuse3_2,filters=256)
    res3_4_2 = resnet(fuse3_2,conv3_4_2,filters=256)
    conv3_4_3 = conv(res3_4_2,filters=256)
    res3_4_3 = resnet(res3_4_2,conv3_4_3,filters=256) 
    
    
    conv4_4_2 = conv(fuse4_2,filters=512)
    res4_4_2 = resnet(fuse4_2,conv4_4_2,filters=512)
    conv4_4_3 = conv(res4_4_2,filters=512)
    res4_4_3 = resnet(res4_4_2,conv4_4_3,filters=512)
    
    
    
    
    
    """reverse stage 3"""
    
    
    conv1_5_1 = fconv(res1_4_3,filters=64)
    conv2_5_1 = fconv(res2_4_3,filters=128)
    conv3_5_1 = fconv(res3_4_3,filters=256)
    
    
    d1_res1_4_3 = dconv(res1_4_3,filters=128)
    d2_res1_4_3 = dconv1(res1_4_3,filters=256)
    
    u1_res2_4_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res2_4_3))
    d1_res2_4_3 = dconv(res2_4_3,filters=256)
    
    u1_res3_4_3 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res3_4_3))
    u2_res3_4_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(res3_4_3))
    
    u1_res4_4_3 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res4_4_3))
    u2_res4_4_3 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(res4_4_3))
    u3_res4_4_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(res4_4_3))
    
    

    fus1_3 = concatenate([conv1_5_1,u1_res2_4_3,u2_res3_4_3,u3_res4_4_3], axis = 3)
    fus2_3 = concatenate([d1_res1_4_3,conv2_5_1,u1_res3_4_3,u2_res4_4_3], axis = 3)
    fus3_3 = concatenate([d2_res1_4_3,d1_res2_4_3,conv3_5_1,u1_res4_4_3], axis = 3)
    
    fuse1_3 = conv(fus1_3,filters=64)
    fuse2_3 = conv(fus2_3,filters=128)
    fuse3_3 = conv(fus3_3,filters=256)

    conv1_5_2 = conv(fuse1_3,filters=64)
    res1_5_2 = resnet(fuse1_3,conv1_5_2,filters=64)
    conv1_5_3 = conv(res1_5_2,filters=64)
    res1_5_3 = resnet(res1_5_2,conv1_5_3,filters=64)
    
    conv2_5_2 = conv(fuse2_3,filters=128)
    res2_5_2 = resnet(fuse2_3,conv2_5_2,filters=128)
    conv2_5_3 = conv(res2_5_2,filters=128)
    res2_5_3 = resnet(res2_5_2,conv2_5_3,filters=128)
    
    conv3_5_2 = conv(fuse3_3,filters=256)
    res3_5_2 = resnet(fuse3_3,conv3_5_2,filters=256)
    conv3_5_3 = conv(res3_5_2,filters=256)
    res3_5_3 = resnet(res3_5_2,conv3_5_3,filters=256)
    
    
 
    
    
    """reverse stage 2"""
    
    
    conv1_6_1 = conv(res1_5_3,filters=64)
    conv2_6_1 = conv(res2_5_3,filters=128)
    
    d1_res1_5_3 = dconv(res1_5_3,filters=128)
    
    u1_res2_5_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res2_5_3))
    
    u1_res3_5_3 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res3_5_3))
    u2_res3_5_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(res3_5_3))

    fus1_4 = concatenate([conv1_6_1,u1_res2_5_3,u2_res3_5_3], axis = 3)
    fus2_4 = concatenate([d1_res1_5_3,conv2_6_1,u1_res3_5_3], axis = 3)
    fuse1_4 = conv(fus1_4,filters=64)
    fuse2_4 = conv(fus2_4,filters=128)
    
    conv1_6_2 = conv(fuse1_4,filters=64)
    res1_6_2 = resnet(fuse1_4,conv1_6_2,filters=64)
    conv1_6_3 = conv(res1_6_2,filters=64)
    res1_6_3 = resnet(res1_6_2,conv1_6_3,filters=64)
    
    
    conv2_6_2 = conv(fuse2_4,filters=128)
    res2_6_2 = resnet(fuse2_4,conv2_6_2,filters=128)
    conv2_6_3 = conv(res2_6_2,filters=128)
    res2_6_3 = resnet(res2_6_2,conv2_6_3,filters=128)
    
    """ reverse stage 1"""
    
    conv1_7_1 = conv(res1_6_3,filters=64)
    u1_res2_6_3 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(res2_6_3))

    fus1_5 =  concatenate([conv1_7_1,u1_res2_6_3], axis = 3)
    fuse1_5 = conv(fus1_5,filters=64)
    
    
    conv1_7_2 = conv(fuse1_5,filters=64)
    res1_7_2 = resnet(fuse1_5,conv1_7_2,filters=64)
    conv1_7_3 = conv(res1_7_2,filters=64)
    res1_7_3 = resnet(res1_7_2,conv1_7_3,filters=64)
    


    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(res1_7_3)

    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    #model = Model(inputs, conv10)
    model = Model(inputs=[inputs,inputs1], outputs=conv10)
    
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=opt, loss = bce_dice_loss, metrics = ['accuracy',dice_loss])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model