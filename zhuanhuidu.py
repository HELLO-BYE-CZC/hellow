# *_*coding:utf-8 *_*
import numpy as np
import sys
import scipy.io as io
import random

import pandas as pd
df =  pd.read_table(r"E:\python\max.txt")
# print(df)
df = pd.DataFrame(df)
# print(df)
# print(df.loc[0])
#print(df.loc[1])


x = df['时间(s) - 曲线 0']
y = df['幅值 - 曲线 0']
x = np.array(x,float)
y = np.array(y)
print(x,y)

normal_imgs = []

max_start = len(y) - 1024
starts = []
for i in range(500):
    # 随机一个start，不在starts里，就加入
    while True:
        start = random.randint(0, max_start)
        if start not in starts:
            starts.append(start)
            break
    temp = y[start: start + 1024]
    temp = np.array(temp)
    temp = temp.reshape(32, 32)
    max = -2
    min = 2
    for i in range(32):
        for j in range(32):
            if (temp[i][j] > max):
                max = temp[i][j]

            if (temp[i][j] < min):
                min = temp[i][j]

    for i in range(32):
        for j in range(32):
            temp[i][j] = 255 * (temp[i][j] - min) / (max - min)

    normal_imgs.append(temp)

# print(normal_imgs)

np.savez("min-max", *normal_imgs)

import numpy as np
from PIL import Image


def load_imgs(npzfile, path, savepath):
    images = np.load(npzfile)
    i = 0
    for file in images:
        i += 1
        image = np.load(path + '/' + file + '.npy')
        image = Image.fromarray(image)
        image = image.convert('L')
        # image.show()
        image.save(savepath + '/' + 'array_%d.jpg' % (i))


load_imgs('max.npz', 'max', 'E:\python\min-max')



import os

path = r"E:\Download\15-2021-10-11" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
txts = []
for file in files: #遍历文件夹
    position = path+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
    print (position)
    with open(position, "r",encoding='utf-8') as f:    #打开文件
        data = f.read()   #读取文件
        txts.append(data)
txts = ','.join(txts)#转化为非数组类型
print (txts)




import scipy.signal as signal
def stft(x, **params):
    '''
    :param x: 输入信号
    :param params: {fs:采样频率；
                    window:窗。默认为汉明窗；
                    nperseg： 每个段的长度，默认为256，
                    noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                    nfft：fft长度，
                    detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                    return_onesided：默认为True，返回单边谱。
                    boundary：默认在时间序列两端添加0
                    padded：是否对时间序列进行填充0（当长度不够的时候），
                    axis：可以不必关心这个参数}
    :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
    '''
    f, t, zxx = signal.stft(x, **params)
    return f, t, zxx
stft(y,fs=100)

import os

path = r"E:\python\xlq\sj"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
txts = []
for file in files:  # 遍历文件夹
    position = path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
    # print (position)

    df = pd.read_table(position)
    # print(df)
    df = pd.DataFrame(df)
    print(df)
    # print(df.loc[0])
    # print(df.loc[1])

    x = df['时间(s) - 曲线 0']
    y = df['幅值 - 曲线 0']
    x = np.array(x, float)
    y = np.array(y)
    print(x, y)
    print(file)


    def stft(x, **params):
        '''
        :param x: 输入信号
        :param params: {fs:采样频率；
                        window:窗。默认为汉明窗；
                        nperseg： 每个段的长度，默认为256，
                        noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                        nfft：fft长度，
                        detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                        return_onesided：默认为True，返回单边谱。
                        boundary：默认在时间序列两端添加0
                        padded：是否对时间序列进行填充0（当长度不够的时候），
                        axis：可以不必关心这个参数}
        :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
        '''
        f, t, zxx = signal.stft(x, **params)
        return f, t, zxx


    stft(y, fs=100)


    def stft_specgram(x, picname=None, **params):  # picname是给图像的名字，为了保存图像
        f, t, zxx = stft(x, **params)
        plt.figure(figsize=(28, 28))
        plt.pcolormesh(t, f, np.abs(zxx))
        # plt.colorbar()
        plt.ylim(0, 50)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        #     if picname is not None:
        #         plt.savefig('..\\picture\\' + str(picname) + '.jpg')       #保存图像
        #     plt.clf()      #清除画布
        plt.savefig('%s.jpg' % file)
        return t, f, zxx


    stft_specgram(y, fs=100, nperseg=128)


import matplotlib.pyplot as plt
def stft_specgram(x, picname=None, **params):    #picname是给图像的名字，为了保存图像
    f, t, zxx = stft(x, **params)
    plt.figure(figsize=(28,28))
    plt.pcolormesh(t, f, np.abs(zxx))
    #plt.colorbar()
    plt.ylim(0,50)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
#     if picname is not None:
#         plt.savefig('..\\picture\\' + str(picname) + '.jpg')       #保存图像
#     plt.clf()      #清除画布
    plt.savefig("examples.jpg")
    return t, f, zxx

stft_specgram(y,fs=100,nperseg=128)