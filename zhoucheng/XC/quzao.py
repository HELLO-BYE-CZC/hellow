# *_*coding:utf-8 *_*
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


# ######读数据
# def shuju():
#     names = os.listdir("E:\\Download\\西储大学轴承数据中心网站\\12k Drive End Bearing Fault Data\\滚动体故障")
#     is_acc = ['mat' in name for name in names]
#     names = names[:sum(is_acc)]
#     print(names)
#     # len(names)
#     for f in range(len(names)):
#         n = names[f]
#
#         n = n.strip('.mat')
#         #     print(n,len(n))
#
#         #     if len(n)==3:
#         d = scio.loadmat('E:\\Download\\西储大学轴承数据中心网站\\12k Drive End Bearing Fault Data\\滚动体故障\\%s.mat' % n)[
#             'X%s_DE_time' % n]
#         np.savetxt("E:\\Download\\12K轴承数据集\\12k\\g\\" + n + ".txt", d)
#     #     else:
#     #         a = "0"+n
#     #         d = scio.loadmat('E:\\Download\\西储大学轴承数据中心网站\\Normal Baseline Data\\%s.mat'%n)['X%s_DE_time'%a]
#     #         np.savetxt("E:\\Download\\西储大学轴承数据中心网站\\Normal Baseline Data\\txt\\"+ n +".txt",d)
#
#

d = np.loadtxt("E:\Download\\12K轴承数据集\\12k\\g\\118.txt")
print(d)
plt.figure(figsize=(12,8))
plt.plot(d)
plt.show()

#
# Fs = 12000     # sampling rate采样率
# Ts = 1.0/Fs    # sampling interval 采样区间
# t = np.arange(len(d))  # time vector,这里Ts也是步长
#
# # ff = 25;     # frequency of the signal
# # y = np.sin(2*np.pi*ff*t)
#
# n = len(d)     # length of the signal
# print(n)
# k = np.arange(n)
# T = n/Fs
# frq = k/T     # two sides frequency range
# frq1 = frq[range(int(n/2))] # one side frequency range
#
# YY = np.fft.fft(d)   # 未归一化
# Y = np.fft.fft(d)/n   # fft computing and normalization 归一化
# Y1 = Y[range(int(n/2))]
# c = frq
#
# x = np.arange(len(d))
#
# y = d
# emd = EMD()
#
# # imfs_emd = emd(y)
# # imfs_eemd = eemd(yy)
# imfs_ceemd = emd(y)
#
# print(np.shape(imfs_ceemd))
#
#
# plt.figure(1,figsize=(12,16))
#
# plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 1 )
# plt.plot(x, y, 'r')
#
# plt.title("Signal Input")
# for i in range(np.shape(imfs_ceemd)[0]):
#     plt.subplot(1 + np.shape(imfs_ceemd)[0],1,2+i)
#     plt.plot(x,imfs_ceemd[i,:],'b')
#     plt.title("IMF-emd"+str(i))
# plt.tight_layout()
# plt.show()
# #
# plt.figure(figsize=(12,16))
# plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 1 )
# plt.plot(frq1,Y1)
# for i in range(np.shape(imfs_ceemd)[0]):
#     plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 2 + i)
#     YY = np.fft.fft(imfs_ceemd[i, :])
#
#     N = len(YY)
#     half_x = x[range(int(N / 2))]  # 取一半区间
#     abs_y = np.abs(YY)  # 取复数的绝对值，即复数的模(双边频谱)
#     normalization_y = abs_y / N  # 归一化处理（双边频谱）
#     yy = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
#
#     plt.plot(frq1,abs_y[range(int(N / 2))], 'b')
# plt.tight_layout()
# plt.show()

N = 12000
y = np.fft.fft(d)
y = abs(y)
y = y/len(y)
print(len(y))
plt.plot(np.arange(int((len(y)/2))),y[range(int((len(y)/2)))])
plt.show()