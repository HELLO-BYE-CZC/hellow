# *_*coding:utf-8 *_*
# *_*coding:utf-8 *_*
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from PyEMD import EMD, CEEMDAN,Visualisation
import pandas as pd
import seaborn


df = pd.read_table(r"E:\python\max.txt")
# print(df)
df = pd.DataFrame(df)
# print(df)
# print(df.loc[0])
# print(df.loc[1])

#x = df['时间(s) - 曲线 0']
y = df['幅值 - 曲线 0']




Fs = 100.0;     # sampling rate采样率
Ts = 1.0/Fs;    # sampling interval 采样区间
t = np.arange(3072)  # time vector,这里Ts也是步长

# ff = 25;     # frequency of the signal
# y = np.sin(2*np.pi*ff*t)

n = len(y)     # length of the signal
print(n)
k = np.arange(n)
T = n/Fs
frq = k/T     # two sides frequency range
frq1 = frq[range(int(n/2))] # one side frequency range

YY = np.fft.fft(y)   # 未归一化
Y = np.fft.fft(y)/n   # fft computing and normalization 归一化
Y1 = Y[range(int(n/2))]
c = frq
#t=x
# t = np.arange(0, 3, 0.01)
# S = np.sin(13*t + 0.2*t**1.4) - np.cos(3*t)
# print(S)
# print(y)
# Extract imfs and residue
# In case of EMD
emd = EMD()
emd.emd(abs(Y))
imfs, res = emd.get_imfs_and_residue()

# In general:
#components = EEMD()(S)
#imfs, res = components[:-1], components[-1]

vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=c, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()

ceemdan = CEEMDAN()
ceemdan.ceemdan(abs(Y))
imfs1, res1 = ceemdan.get_imfs_and_residue()
vis = Visualisation()
vis.plot_imfs(imfs=imfs1, residue=res1, t=c, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs1)
vis.show()