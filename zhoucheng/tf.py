import os
import scipy.io
from matplotlib import pyplot as plt
from PIL import Image
from scipy.linalg import logm
import numpy as np
import pandas as pd
from PyEMD import EMD,CEEMDAN,Visualisation,EEMD
import math
from pandas import Series
from scipy.signal import argrelextrema
from IPython.display import clear_output


path = 'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\'


def get_a_bearings_data(folder):
    ''' 获取某个工况下某个轴承的全部n个csv文件中的数据，返回numpy数组
    dp:bearings_x_x的folder
    return:folder下n个csv文件中的数据，shape:[n*32768,2]=[文件个数*采样点数，通道数]'''
    names = os.listdir(folder)
    is_acc = ['acc' in name for name in names]
    names = names[:sum(is_acc)]
    files = [os.path.join(folder, f) for f in names]
    # Bearing1_4 的csv文件的分隔符是分号：';'
    print(pd.read_csv(files[0], header=None).shape)
    sep = ';' if pd.read_csv(files[0], header=None).shape[-1] == 1 else ','
    h = [pd.read_csv(f, header=None, sep=sep).iloc[:, -2] for f in files]
    v = [pd.read_csv(f, header=None, sep=sep).iloc[:, -1] for f in files]
    H = np.concatenate(h)
    V = np.concatenate(v)
    print(H.shape, V.shape)
    return np.stack([H, V], axis=-1)


# p = 'E:\\download\\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\\phm-ieee-2012-data-challenge-dataset'
#
# for i in ['Learning_set', 'Full_Test_Set']:
#     pp = os.path.join(p, i)
#     for j in os.listdir(pp):
#         ppp = os.path.join(pp, j)
#         print(ppp)
#
#         data = get_a_bearings_data(ppp)
#         save_name = p + '\\mat\\' + j + '.mat'
#         print(save_name)
#         scipy.io.savemat(save_name, {'h': data[:, 0], 'v': data[:, 1]})  # 写入mat文件
#     print('\n')

p = 'E:\\download\\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\\phm-ieee-2012-data-challenge-dataset\\Full_Test_Set\\Bearing3_3\\'


data = get_a_bearings_data(p)