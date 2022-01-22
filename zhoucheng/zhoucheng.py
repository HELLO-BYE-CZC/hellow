# *_*coding:utf-8 *_*
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
# path = 'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\'
'读取某工况下某个轴承的某个采样数据csv文件，并观察'
'''d = pd.read_csv('E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\Learning_set\Bearing1_1\\acc_00001.csv',
                header=None, sep=',')

plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.plot(d.iloc[:, -2])
plt.title('Horizontal_vibration_signals')
plt.subplot(122)
plt.plot(d.iloc[:, -1])
plt.title('Vertical_vibration_signals')
plt.show()'''

#
# def get_a_bearings_data(folder):
#     ''' 获取某个工况下某个轴承的全部n个csv文件中的数据，返回numpy数组
#     dp:bearings_x_x的folder
#     return:folder下n个csv文件中的数据，shape:[n*32768,2]=[文件个数*采样点数，通道数]'''
#     names = os.listdir(folder)
#     is_acc = ['acc' in name for name in names]
#     names = names[:sum(is_acc)]
#     files = [os.path.join(folder, f) for f in names]
#     # Bearing1_4 的csv文件的分隔符是分号：';'
#     print(pd.read_csv(files[0], header=None).shape)
#     sep = ';' if pd.read_csv(files[0], header=None).shape[-1] == 1 else ','
#     h = [pd.read_csv(f, header=None, sep=sep).iloc[:, -2] for f in files]
#     v = [pd.read_csv(f, header=None, sep=sep).iloc[:, -1] for f in files]
#     H = np.concatenate(h)
#     V = np.concatenate(v)
#     print(H.shape, V.shape)
#     return np.stack([H, V], axis=-1)
#
#
# p = 'E:\\download\\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\\phm-ieee-2012-data-challenge-dataset'

# # for i in ['Learning_set', 'Full_Test_Set']:
# #     pp = os.path.join(p, i)
# #     for j in os.listdir(pp):
# #         ppp = os.path.join(pp, j)
# #         print(ppp)
# #
# #         data = get_a_bearings_data(ppp)
# #         save_name = p + '\\mat\\' + j + '.mat'
# #         print(save_name)
# #         scipy.io.savemat(save_name, {'h': data[:, 0], 'v': data[:, 1]})  # 写入mat文件
# #     print('\n')
#
# p = 'E:\\download\\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\\phm-ieee-2012-data-challenge-dataset\\Full_Test_Set\\Bearing3_3\\'
#
#
# data = get_a_bearings_data(p)
#
#
# np.savetxt('E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\Bearing3_3.txt',data)  # 写入mat文件
#


d = pd.read_csv('E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\Bearing3_3.txt',
                header=None, sep=' ')

# print(d)
# plt.figure(figsize=(20, 5))
# plt.subplot(121)
# plt.plot(d.iloc[:, 0])
# plt.title('Horizontal_signals')
# plt.subplot(122)
# plt.plot(d.iloc[:, 1])
# plt.title('Vertical_signals')
# plt.show()
#



#
# d = d.iloc[:,1]
# d = np.array(d)
#
# #3文件等分1085个
#
#
# def dengfen():
#     n = 0
#     for i in range(1085):
#
#         s = n
#         n = n + 1024
#         a = d[s:n]
#         p = r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3'
#         np.savetxt(p+'\Bearing3_3_' +str(i)+'.txt',a)
#
# dengfen()
#
#
# # 看分解的一个信号
# p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_0.txt',
#                 header=None,sep=' ')
# plt.figure(figsize=(20, 5))
#
# plt.plot(p)
# plt.title('Horizontal_signals')
# plt.show()





#计算峰峰值
# def fengfengzhi():
#     feng = []
#     for i in range(1085):
#         p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_'+str(i)+'.txt',header=None)
#         p.stack().max()
#         p.stack().min()
#         a = p.stack().max() - p.stack().min()
#         feng.append(a)
#     print(feng)
#     plt.figure(figsize=(5, 5))
#
#     plt.plot(feng)
#     plt.title('fengfengzhi')
#     plt.show()
#     # print(p.shape)
#     # np.savetxt(path+'fengfengzhi.txt',feng)
# fengfengzhi()


#计算均方根
# def junfanggen():
#     junfang = []
#     for i in range(1085):
#         p = pd.read_csv(
#             r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_' + str(
#                 i) + '.txt',header=None)
#         p = np.array(p)
#         a =0
#         for j in range(len(p)):
#             a = a + math.pow(p[j][0],2)
#         a = a/len(p)
#         a = np.sqrt(a)
#         junfang.append(a)
#     plt.figure(figsize=(5, 5))
#     plt.plot(junfang)
#     plt.title('junfanggen')
#     plt.show()
#     # np.savetxt(path + 'junfanggen.txt', junfang)
# junfanggen()

# def qiaodu():
#     qiaodu = []
#     for i in range(1085):
#         p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_'+str(i)+'.txt',header=None)
#         a = p.kurt()
#         qiaodu.append(a)
#     print(type(a))
#
#     plt.figure(figsize=(5, 5))
#
#     plt.plot(qiaodu)
#     plt.title('qiaodu')
#     plt.show()
#     print(p.shape)
#     # np.savetxt(path + 'qiaodo.txt', qiaodu)
# qiaodu()

# def boxingyinzi():
#     boxing = []
#     for i in range(1085):
#         p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_'+str(i)+'.txt',header=None)
#         p = np.array(p)
#         a =0
#         for j in range(len(p)):
#             a = a + math.pow(p[j][0],2)
#         a = a/len(p)
#         a = np.sqrt(a)
#         a = a / (abs(p).mean())
#         boxing.append(a)
#
#     plt.figure(figsize=(5, 5))
#
#     plt.plot(boxing)
#     plt.title('boxing')
#     plt.show()
#     # print(p.shape)
#     # np.savetxt(path + 'boxingyinzi.txt', boxing)
# boxingyinzi()



# def yuduyinzi():
#     yudu = []
#     for i in range(1085):
#         p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_'+str(i)+'.txt',header=None)
#         summ = 0
#         p = np.array(p)
#         for j in range(1024):
#             summ =summ+ math.sqrt(abs(p[j][0]))
#         a = p.max() / math.pow((summ / len(p)), 2)
#         yudu.append(a)
#     plt.figure(figsize=(5, 5))
#
#     plt.plot(yudu)
#     plt.title('yudu')
#     plt.show()
#     # print(p.shape)
#     # np.savetxt(path + 'yuduyinzi.txt', yudu)
# yuduyinzi()



# pstf_list=[]
# def psfeatureTime(data,p1,p2):
#
#
#      #均值
#      df_mean=data[p1:p2].mean()
#      #方差
#      df_var=data[p1:p2].var()
#      #标准差
#      df_std=data[p1:p2].std()
#      #均方根
#      df_rms=math.sqrt(pow(df_mean,2) + pow(df_std,2))
#      #偏度
#      df_skew=data[p1:p2].skew()
#      #峭度
#      df_kurt=data[p1:p2].kurt()
#      sum=0
#      for i in range(p1,p2):
#       sum+=math.sqrt(abs(data[i]))
#      #波形因子
#      df_boxing=df_rms / (abs(data[p1:p2]).mean())
#      #峰值因子
#      df_fengzhi=(max(data[p1:p2])) / df_rms
#      #脉冲因子
#      df_maichong=(max(data[p1:p2])) / (abs(data[p1:p2]).mean())
#      #裕度因子
#      df_yudu=(max(data[p1:p2])) / pow((sum/(p2-p1)),2)
#      featuretime_list = [df_mean,df_rms,df_skew,df_kurt,df_boxing,df_fengzhi,df_maichong,df_yudu]
#      return featuretime_list

'''def zhibiao():
    pstf_list = []
    for i in range(1085):
        p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_'+str(i)+'.txt',header=None)
        df_ffz = p.stack().max() - p.stack().min()
        # 均值
        df_mean = p.mean()
        # 方差
        df_var = p.var()
        # 标准差
        df_std = p.std()
        # 均方根
        df_rms = math.sqrt(pow(df_mean, 2) + pow(df_std, 2))
        # 峭度
        df_kurt = p.kurt()
        # 波形因子
        df_boxing = df_rms / (abs(p).mean())
        # 裕度因子
        summ = 0
        peak = p.stack().max()
        p = np.array(p)
        for j in range(1024):
            summ =summ+ math.sqrt(abs(p[j][0]))
        df_yudu = (peak) / math.pow((summ / len(p)), 2)
        featuretime_list = [df_ffz, df_rms, df_kurt, df_boxing, df_yudu]
        print(featuretime_list)
    return pstf_list
zhibiao()
#print(pstf_list)'''


'''# def nengliangshang():
#     Hn = []
#     for i in range(1085):
#         p = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_'+str(i)+'.txt',header=None)
#         p = np.array(p)
#         a = []
#         for k in range(len(p)):
#             b = p[k][0]
#             a.append(b)
#
#         x = np.arange(len(p))
#         # print(len(a))
#         a = np.array(a)
#         # print(a)
#         eemd = EEMD()
#         imfs_eemd = eemd(a)
#         # print(np.shape(imfs_eemd))
#         E1 = 0
#         h = 0
#         for j in range(np.shape(imfs_eemd)[0]):
#
#             a = imfs_eemd[j,:]
#             Ei = a**2
#             E =sum(Ei)
#             EE = E1 +E
#             E1 = EE
#             pi = E/EE
#             H = pi*np.log2(pi)
#             H1 = h +H
#             h = H1
#         H = -H1
#         Hn.append(H)
#         print('正在打印第%d个'%i)
#     print(Hn)
#     np.savetxt(
#         'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\nengliangshang.txt',Hn)
# nengliangshang()'''       #错能量熵

#能量熵
# def nengliangshang():
#     H = []
#     for i in range(1085):
#         p = np.loadtxt(
#             r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\Bearing3_3\Bearing3_3_' + str(
#                 i) + '.txt')
#         a = np.array(p)
#         # print(a)
#         eemd = EEMD(trials=10)
#         imfs_eemd = eemd(a)
#         # print(np.shape(imfs_eemd))
#         Ei = []
#         for s in range(4):
#             a = imfs_eemd[s, :]
#             # print(len(a))
#             E = a ** 2
#             E = sum(E)
#             Ei.append(E)
#         # print(Ei)
#         E = sum(Ei)
#         # print(E)
#         pi = Ei / E
#         # print(pi,type(pi))
#         # print(np.log2(pi),np.log10(pi))
#         # print(pi[0],np.log2(pi[0]),np.log10(pi[0]))
#         # print(pi * np.log10(pi))
#         h = -sum(pi * np.log2(pi))
#         H.append(h)
#         print('正在打印第%d个' % i)
#     np.savetxt(
#         'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\\txt\\4nengliangshang.txt',
#         H)
# nengliangshang()


# n= pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\PCAguiyihua.txt',header=None)
#
# # n = np.array(n)
# # gyh = (n-np.min(n))/(np.max(n)-np.min(n))
# # np.savetxt(path + 'nengliangshangguiyihua.txt',gyh)
# plt.figure(figsize=(5, 5))
# plt.plot(n)
# # plt.plot(n,'b')
# plt.title('PCAguiyihua')
# plt.show()
# #
#

import pandas as pd
import numpy as np
from PyEMD import EMD, CEEMDAN, EEMD
from PIL import Image

path = "E:\\Download\\12K轴承数据集\\12k\\o\\fj\\"
names = os.listdir(path)
for q in range(len(names)):
    n = names[q]
    SJ = np.loadtxt(path + n)
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
            x1 = pd.Series(SJ)
            y1 = pd.Series(imfs_emd[i, :])
            cor = round(x1.corr(y1), 4)
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
    photo.save("E:\\Download\\12K轴承数据集\\12k\\o\\tp\\" + str(q) + '.jpg')