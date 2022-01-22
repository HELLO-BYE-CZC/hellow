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


# path = 'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\'
# ffz = np.loadtxt(path+'fengfengzhi.txt')
# jfg = np.loadtxt(path+'junfanggen.txt')
# qd = np.loadtxt(path+'qiaodo.txt')
# bx = np.loadtxt(path+'boxingyinzi.txt')
# yd = np.loadtxt(path+'yuduyinzi.txt')
# gyh = np.loadtxt(path + 'nengliangshangguiyihua.txt')


path = 'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\'
ffz = pd.read_csv(path+'fengfengzhi.txt',header=None)
jfg = pd.read_csv(path+'junfanggen.txt',header=None)
qd = pd.read_csv(path+'qiaodo.txt',header=None)
bx = pd.read_csv(path+'boxingyinzi.txt',header=None)
yd = pd.read_csv(path+'yuduyinzi.txt',header=None)
nls = pd.read_csv(path+'4nengliangshang.txt',header=None)
gyh = pd.read_csv(path + 'nengliangshangguiyihua.txt',header=None)


def autoLinNorm(data):  # 传入一个矩阵
    ''' 0-1归一化
        :param data: []矩阵
        :return:     []
    '''
    mins = data.min(0)  # 返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)  # 返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins  # 最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))  # 生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]  # 返回 data矩阵的行数
    normData = data - np.tile(mins, (row, 1))  # data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges, (row, 1))  # data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData


# zb5 = pd.DataFrame[ffz,jfg,qd,bx,yd]
zb5 = pd.concat([ffz,jfg,qd,bx,yd],axis=1)
zb5 = np.array(zb5)


zb6 = pd.concat([ffz,jfg,qd,bx,yd,nls],axis=1)
zb6 = np.array(zb6)
np.savetxt(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\zb6.txt',zb6)


zb6 = autoLinNorm(zb6)
# print(autoLinNorm(zb5))

# gyh = pd.array((zb5-np.min(zb5))/(np.max(zb5)-np.min(zb5)),axis=0)
# print(gyh)
#
#
from sklearn.decomposition import PCA
# X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
# print(X.shape,zb5.shape)
pca = PCA(n_components=1)   #降到2维
# pca.fit(zb5)                  #训练
newX=pca.fit_transform(zb6)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)                  #输出降维后的数据

newX = autoLinNorm(newX)
# np.savetxt(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\PCAguiyihua.txt',newX)
plt.figure()
plt.plot(newX)
plt.show()