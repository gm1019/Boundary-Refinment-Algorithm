# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:51:19 2024

@author: Ming Gong
"""

from hrunet import *
from data3 import *
import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping
import os
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img




os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(16,'data/refine/liver','img','pred','binary',data_gen_args,save_to_dir = None)
#myGen = validGenerator(64,'data/refine','image','maskt',data_gen_args,save_to_dir = None)



model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='loss', patience=10,mode='min')
#lr_scheduler = LearningRateScheduler(lr_schedule)
#lr_reducer = ReduceLROnPlateau(factor=0.5,
#                              cooldown=0,
#                              patience=2,


#                              mode='min',
#                               min_lr=1e-5)
 
 #callbacks = [model_checkpoint, lr_reducer, lr_scheduler]
#model.fit_generator(myGene,steps_per_epoch=200,epochs=20,callbacks=callbacks)


#model.fit_generator(myGene,steps_per_epoch=1800,epochs=150,validation_data=myGen,
#                    validation_steps=100,callbacks=[early_stopping])
model.fit_generator(myGene,steps_per_epoch=30,epochs=1,callbacks=[early_stopping])
model.save('data/membrane/predict/my_unetr1.h5')




target_dir = 'data/refine/patch/img/'
test_img_paths = sorted([os.path.join(target_dir, fname)
                           for fname in os.listdir(target_dir)
                           if fname.endswith(".png") and not fname.startswith(".")])

target_dir1 = 'data/refine/patch/pred/'
mk_paths = sorted([os.path.join(target_dir1, fname)
                           for fname in os.listdir(target_dir1)
                           if fname.endswith(".png") and not fname.startswith(".")])

testGen1 = testGenerator(test_img_paths, mk_paths, num_image=7253)
results1 = model.predict(testGen1,7253,verbose=1)

saveResult("data/refine/predict/p4",results1,test_img_paths)


"""
target_dir = 'data/refine/ss/img/'
test_img_paths = sorted([os.path.join(target_dir, fname)
                           for fname in os.listdir(target_dir)
                           if fname.endswith(".png") and not fname.startswith(".")])

target_dir1 = 'data/refine/ss/mask/'
mk_paths = sorted([os.path.join(target_dir1, fname)
                           for fname in os.listdir(target_dir1)
                           if fname.endswith(".png") and not fname.startswith(".")])


testGen1 = testGenerator(test_img_paths,mk_paths,num_image=14682)
results1 = model.predict_generator(testGen1,14682,verbose=1)

saveResult("data/membrane/predict/p13",results1,test_img_paths)
"""




