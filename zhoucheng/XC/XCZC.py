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
# ###########################等分数据
# def dengfen(SJ):
#     n = 0
#     tupian = []
#     for f in range(64):
#         s = n
#         n = n + 128
#         a = SJ[s:n]
#         emd = EMD()
#         imfs_emd = emd(a)
#         xg = []
#         for i in range(imfs_emd.shape[0]):
#             #         print(imfs_emd[i,:])
#             x1 = pd.Series(n)
#             y1 = pd.Series(imfs_emd[i, :])
#             cor = round(x1.corr(y1), 4)
#             xg.append(cor)
#             xgmax = xg.index(max(xg))
#             xgfft = imfs_emd[xgmax,:]
#             xgfft = np.fft.fft(xgfft)
#             xgfft = abs(xgfft)
#         tupian.append(xgfft)
#     return  tupian
# # shuju()
#
# #####分解8172
# def fj():
#     path = "E:\\Download\\12K轴承数据集\\12k\\g"
#     names = os.listdir(path)
#
#     len(names)
#     for f in range(len(names)):
#         d = np.loadtxt("E:\\Download\\12K轴承数据集\\12k\\g\\%s" % names[f])
#         z = len(d) // 8172
#         n = 0
#         print(z)
#         for i in range(z):
#             s = n
#             n = n + 8192
#             a = d[s:n]
#             np.savetxt(path + '\\fj\\%s' % f + str(i) + '.txt', a)
#
# # fj()
#
#

#

from PyEMD import EMD, CEEMDAN, EEMD
from PIL import Image
# ##########8172分64+相关系数 + fft
def fen64():
    path = "E:\\Download\\12K轴承数据集\\12k\\n\\fj\\"
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
                '''x1 = pd.Series(SJ)
                y1 = pd.Series(imfs_emd[i, :])
                cor = round(x1.corr(y1), 4)'''

                x1 = a
                y1 = imfs_emd[i, :]
                z1 = x1 * y1
                z1 = np.sum(z1)
                cor = z1 / (np.sqrt(np.sum(x1 ** 2) * np.sum(y1 ** 2)))  #相关系数


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
        photo.save("E:\\Download\\12K轴承数据集\\tftp\\ntp\\" + str(q) + '.png')


# fen64()





path = 'E:\\Download\\12K轴承数据集\\tftp\\'

def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    #print(cate)
    # for x in os.listdir(path):
    #     #print(x)
    #     cate.append(path + x)
    #print(cate)
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.png'):
#             print('reading the images:%s' % (im))
            img =io.imread(im)
#             print(img.shape)
            #img = transform.resize(img, (w, h))
            # print(img.shape)
            imgs.append(img)
            labels.append(idx)
#         print(labels)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)

num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]/255

label = label[arr]
ratio = 0.7
s = np.int_(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_test = data[s:]
y_test = label[s:]


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='sigmoid')
        self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y
model = LeNet5()

from keras import optimizers
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])




history = model.fit(x_train, y_train, batch_size=8, epochs=20, validation_split=0.1, validation_freq=1,
                    )
model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()