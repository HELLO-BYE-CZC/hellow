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
import re

# print(pattern.search(str))
path = "E:\CAET\有车"
# ######读数据
def dushuju():

    names = os.listdir(path)
    # names.sort(key=lambda x: x.split('.')[0])
    # print(names)

    # names = names[0]
    print(names)
    l = []
    for q in range(len(names)):
        # a = names[q]
        # # if a.startswith('1'):
        # #     d = re.findall('1-%d'%(q+1)+'.*',a)
        # #     print(d)
        # #     # print(a[-24:])
        # print(a,a[3])
        # a = a[2]+'.txt'
        # print(a)
        a = pd.read_csv(path+ '\\'+names[q])
        a = np.array(a)
        a = a[9]
        l.append(a)
        # print(a)
    plt.plot(l)
    plt.show()
    # len(names)
    # for q in range(len(names)):
    #     n = names[q]
    #     files = os.path.join(path, n)
    #     files = files.strip()
    #     print(files)
    #     print(files.isspace())

dushuju()
