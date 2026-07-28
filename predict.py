# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:36:32 2024

@author: Ming Gong
"""


# -*- coding: utf-8 -*-

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
from skimage import io, transform
from skimage.io import imsave


os.environ["CUDA_VISIBLE_DEVICES"] = "0"








#model = load_model('data/membrane/predict/thesis/chapter6_1a.h5',custom_objects={'bce_dice_loss': bce_dice_loss,'dice_loss': dice_loss}) 
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model = load_model('data/membrane/predict/my_unetr.h5',custom_objects={'bce_dice_loss': bce_dice_loss,'dice_loss': dice_loss}) 




target_dir = 'data/refine/patch/img/'
mask_dir = 'data/refine/patch/pred/'

def load_test_images(test_img_paths, target_size=(64, 64), as_gray=True):
    images = []
    for path in test_img_paths:
        img = io.imread(path, as_gray=as_gray)
        img = img / 255
        img = transform.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))  # 将图像变为 (64, 64, 1)
        images.append(img)
    return np.array(images)  # 返回形状为 (num_images, 64, 64, 1) 的数组

# 加载掩码图像
def load_test_masks(mk_paths, target_size=(64, 64), as_gray=True):
    masks = []
    for path in mk_paths:
        mask = io.imread(path, as_gray=as_gray)
        mask = mask / 255
        mask = transform.resize(mask, target_size)
        mask = np.reshape(mask, mask.shape + (1,))
        masks.append(mask)
    return np.array(masks)


def save_predicted_images(predictions, save_path, test_img_paths):
    """
    保存预测结果为图像文件
    :param predictions: 模型的预测输出 (num_images, 64, 64, 1)
    :param save_path: 预测结果保存的路径
    :param test_img_paths: 原始测试图像路径列表，用于命名预测结果
    """
    # 如果保存目录不存在，创建它
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 遍历每个预测结果并保存为图像
    for i, pred in enumerate(predictions):
        # 获取图像名
        img_name = os.path.basename(test_img_paths[i])
        
        
        # 对预测结果进行阈值处理，假设0.5为阈值
        pred = (pred > 0.5).astype(np.uint8)  # 将概率值二值化为0或1
        
        # 去掉最后的通道维度，将形状变为 (64, 64)
        pred = np.squeeze(pred, axis=-1)
        
        # 构造保存路径
        save_file_path = os.path.join(save_path, img_name)
        
        # 保存预测图像
        imsave(save_file_path, pred)
        print(f"Saved: {save_file_path}")


test_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".png")])
mk_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith(".png")])

# 加载所有测试图像和掩码为 NumPy 数组
X_test = load_test_images(test_img_paths)
X_test1 = load_test_masks(mk_paths)

# 确认加载的数据形状
print(f"X_test shape: {X_test.shape}, X_test1 shape: {X_test1.shape}")

# 使用模型进行预测，传入 [X_test, X_test1]
results = model.predict([X_test, X_test1], verbose=1)
save_path = 'data/membrane/predict/p13'

save_predicted_images(results,save_path,test_img_paths)


"""

filenames = generate_filenames_from_folder('data/membrane/crop_labelt', extension='.png')

testGen1 = testGenerator("data/membrane/crop_labelt",filenames)

results1 = model.predict_generator(testGen1,4,verbose=1)

saveResult("data/membrane/predict/p4",results1,filenames)








target_dir = 'data/refine/ssss/img/'
test_img_paths = sorted([os.path.join(target_dir, fname)
                           for fname in os.listdir(target_dir)
                           if fname.endswith(".png") and not fname.startswith(".")])
testGen1 = testGenerator(test_img_paths,num_image=7253)
results1 = model.predict_generator(testGen1,7253,verbose=1)

saveResult("data/membrane/predict/p4",results1,test_img_paths)





target_dir = 'data/refine/ssss/img/'
test_img_paths = sorted([os.path.join(target_dir, fname)
                           for fname in os.listdir(target_dir)
                           if fname.endswith(".png") and not fname.startswith(".")])

target_dir1 = 'data/refine/ssss/mask/'
mk_paths = sorted([os.path.join(target_dir1, fname)
                           for fname in os.listdir(target_dir1)
                           if fname.endswith(".png") and not fname.startswith(".")])




#testGen1 = testGenerator(test_img_paths,mk_paths,num_image=100)
results1 = model.predict_generator(combined_gen,100,verbose=1)

saveResult("data/membrane/predict/p13",results1,test_img_paths)
"""