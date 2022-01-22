# *_*coding:utf-8 *_*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.io as scio
from PyEMD import EMD,CEEMDAN,EEMD
import os
from scipy.io import loadmat
import tensorflow as tf
import os
import sys
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import glob
from skimage import transform
from PIL import Image
import skimage
from skimage import io
path = "E:\CAET\有车"
# ######读数据
def dushuju():

    names = os.listdir(path)
    print(names)

    # len(names)
    for q in range(len(names)):
        n = names[q]
        files = os.path.join(path, n)
        files = files.strip()
        print(files)
        print(files.isspace())

        # SJ = np.loadtxt(files)

        SJ = pd.read_csv(files)
        SJ = SJ[10:]
        print(type(SJ))
        SJ =np.array(SJ).reshape(-1,1)
        SJ = SJ.tolist()
        list =[]
        for l in range(len(SJ)):
            list.append(SJ[l][0])
        SJ = list
        SJ = np.array(SJ)
        print(SJ)
        n = 0
        tupian = []
        for f in range(64):
            s = n
            n = n + 128
            a = SJ[s:n]
            #         print(a)
            emd = EMD()
            imfs_emd = emd(a)
            xg = []
            for i in range(imfs_emd.shape[0]):
                #         print(imfs_emd[i,:])
                '''x1 = pd.Series(SJ)
                y1 = pd.Series(imfs_emd[i, :])
                cor = round(x1.corr(y1), 4)'''

                x1 = a
                y1 = imfs_emd[i, :]
                z1 = x1 * y1
                z1 = np.sum(z1)
                cor = z1 / (np.sqrt(np.sum(x1 ** 2) * np.sum(y1 ** 2)))  # 相关系数

                xg.append(cor)
            #         print(xg)
            xgmax = xg.index(max(xg))
            #         print(xgmax)
            xgfft = imfs_emd[xgmax, :]
            xgfft = np.fft.fft(xgfft)
            xgfft = abs(xgfft)
            #         print(xgfft,len(xgfft))
            xgfft = xgfft[0:64]
            #         print(xgfft)
            gy = 255 * (xgfft - np.min(xgfft)) / (np.max(xgfft) - np.min(xgfft))
            tupian.append(gy)
        #     print(tupian)
        tupian = np.array(tupian)
        #     print (tupian.shape)

        im = Image.fromarray(tupian)
        im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        threshold = 127  # 设定的阈值
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)
        photo = im.point(table, '1')
        # photo.show()
        photo.save("E:\\CAET\\有车图片\\" + str(q) + '.png')


dushuju()

# SJ = np.loadtxt(files)

# SJ = pd.read_csv('E:\python\min_max_all\min.txt',sep='\t')
# print(SJ)
# SJ = SJ['幅值 - 曲线 0']
# print(type(SJ))
# SJ =np.array(SJ).reshape(-1,1)
# SJ = SJ.tolist()
# list =[]
# for l in range(len(SJ)):
#     list.append(SJ[l][0])
# SJ = list
# SJ = np.array(SJ)
# print(SJ)
# n = 0
# tupian = []
# for f in range(50):
#     s = n
#     n = n + 100
#     a = SJ[s:n]
#     #         print(a)
#     emd = EMD()
#     imfs_emd = emd(a)
#     xg = []
#     for i in range(imfs_emd.shape[0]):
#         #         print(imfs_emd[i,:])
#         '''x1 = pd.Series(SJ)
#         y1 = pd.Series(imfs_emd[i, :])
#         cor = round(x1.corr(y1), 4)'''
#
#         x1 = a
#         y1 = imfs_emd[i, :]
#         z1 = x1 * y1
#         z1 = np.sum(z1)
#         cor = z1 / (np.sqrt(np.sum(x1 ** 2) * np.sum(y1 ** 2)))  # 相关系数
#
#         xg.append(cor)
#     #         print(xg)
#     xgmax = xg.index(max(xg))
#     #         print(xgmax)
#     xgfft = imfs_emd[xgmax, :]
#     xgfft = np.fft.fft(xgfft)
#     xgfft = abs(xgfft)
#     #         print(xgfft,len(xgfft))
#     xgfft = xgfft[0:50]
#     #         print(xgfft)
#     gy = 255 * (xgfft - np.min(xgfft)) / (np.max(xgfft) - np.min(xgfft))
#     tupian.append(gy)
# #     print(tupian)
# tupian = np.array(tupian)
# #     print (tupian.shape)
#
# im = Image.fromarray(tupian)
# im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
# threshold = 127  # 设定的阈值
# table = []
# for i in range(256):
#     if i < threshold:
#         table.append(0)
#     else:
#         table.append(1)
# photo = im.point(table, '1')
# # photo.show()
# photo.save("E:\\CAET\\无车图片\\" + str(0) + '.png')