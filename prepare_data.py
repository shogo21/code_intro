import cv2
import csv
import os
import numpy as np
import glob
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from PIL import Image
import random

#ビデオのパスのリストを作成する関数
def videotopath(videos):
    x_paths = []
    x_labels = []
    for file in videos:
        hands = sorted(glob.glob(file+"/*"))
        for finger in ['index', 'middle', 'ring', 'little']:
            x_path = list(hand for hand in hands if finger in hand)
            x_label = list(map(lambda p: (1 if "touch" in p else 0), x_path))
            x_paths.append(x_path)
            x_labels.append(x_label)
    #リストの長さが違うからdtype=objectを指定する
    x_paths = np.asarray(x_paths, dtype="object")
    x_labels = np.asarray(x_labels, dtype="object")

    return x_paths, x_labels

#リストの要素のパスの画像を読み込む関数
def imageread(paths):
    inputs = []
    for i, path in enumerate(paths):
        input = np.array([np.array(Image.open(png)) for png in path])
        input = input.astype('float32') / 255
        inputs.append(input)

        if (i % 50) == 0:
            print(str(i)+"番目終了")
    return inputs

#read gazou resize
def imageread2(paths):
    inputs = []
    for i, path in enumerate(paths):
        input = []
        for png in path:
            gazou = Image.open(png)
            gazou = gazou.resize((64, 64))
            input.append(np.asarray(gazou))
        input = np.asarray(input)
        input = input.astype('float32') / 255
        inputs.append(input)

        if (i % 50) == 0:
            print(str(i)+"番目終了")
    return inputs

#changebright
def changebright(ch_lists):
    change_inputs= []
    for ch_list in ch_lists:
        #bright = random.randrange(3, 20, 1) / 10
        change_input1 = cv2.convertScaleAbs(ch_list*255, alpha=0.5, beta=0)
        #change_input2 = cv2.convertScaleAbs(ch_list*255, alpha=1.5, beta=0)
        """change_input3 = cv2.convertScaleAbs(ch_list*255, alpha=0.3, beta=0)
        change_input4 = cv2.convertScaleAbs(ch_list*255, alpha=2.0, beta=0)
        change_input5 = cv2.convertScaleAbs(ch_list*255, alpha=0.8, beta=0)
        change_input6 = cv2.convertScaleAbs(ch_list*255, alpha=1.2, beta=0)"""
        change_input1 = np.asarray(change_input1).astype('float32') / 255
        #change_input2 = np.asarray(change_input2).astype('float32') / 255
        """change_input3 = np.asarray(change_input3).astype('float32') / 255
        change_input4 = np.asarray(change_input4).astype('float32') / 255
        change_input5 = np.asarray(change_input5).astype('float32') / 255
        change_input6 = np.asarray(change_input6).astype('float32') / 255"""
        change_inputs.append(change_input1)
        #change_inputs.append(change_input2)
        """change_inputs.append(change_input3)
        change_inputs.append(change_input4)
        change_inputs.append(change_input5)
        change_inputs.append(change_input6)"""
        #print(bright)
        """change_input = cv2.convertScaleAbs(ch_list*255, alpha=bright, beta=0)
        change_input = np.asarray(change_input).astype('float32') / 255
        change_inputs.append(change_input)"""

    return change_inputs

#パディングを行う関数
def padhing(pa_list):
    #シーケンスをパディング（ゼロパディング）し、同じ長さに揃える
    padded = pad_sequences(pa_list, padding='post', dtype='float32')
    #マスクテンソルを生成
    mask = (padded != 0).astype('float32')
    sequence =padded * mask

    return sequence

#正解ラベルのリストを作成する関数
def correctlabel(co_labels):
    Y = []
    for label in co_labels:
        y = np.array(label)
        y = y.reshape(-1, 1)
        """for j in range(3):
            Y.append(y)"""
        Y.append(y)

    return Y

#AdditionCorrectLabel
def addcorrectlabel(colabels):
    add_Y = []
    for label in colabels:
        add_y = np.array(label)
        add_y = add_y.reshape(-1, 1)
        """for j in range(1):
            add_Y.append(add_y)"""

    return add_Y

#physical_devices = tf.config.list_physical_devices('GPU')
#if len(physical_devices) > 0:
   #for device in physical_devices:
        #tf.config.experimental.set_memory_growth(device, True)
       # print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
#else:
    #print('Not enough GPU hardware devices available')