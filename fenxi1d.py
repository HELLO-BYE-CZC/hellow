# *_*coding:utf-8 *_*
import tensorflow as tf  # 将tensorflow命名为tf
from keras.models import Sequential  # 导入序列函数
from keras.wrappers.scikit_learn import KerasClassifier  # 导入分类标签显示工具
from keras.utils import np_utils  # 导入独热吗部分编辑函数
from sklearn.model_selection import cross_val_score, train_test_split, KFold  # 导入独热吗部分编辑函数
from sklearn.preprocessing import LabelEncoder  # 导入独热吗部分编辑函数
from keras.layers import Dense, SeparableConv1D, LSTM, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D, \
    BatchNormalization, GlobalAveragePooling1D  # 导入卷积层、全连接层、等等
from keras.models import load_model  # 导入保存模型工具
from keras.models import model_from_json  # 导入保存模型工具
import matplotlib.pyplot as plt  # 导入画图工具
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵,这段代码用不上
import itertools  # 这个我忘记了
from keras import layers  # 导入层

import numpy as np
import scipy.io as scio
import pandas as pd


df =  pd.read_table(r"E:\python\max.txt")
# print(df)
df = pd.DataFrame(df)
# print(df)
# print(df.loc[0])
#print(df.loc[1])

y1 = df['幅值 - 曲线 0']
data1 = np.array(y1)
data1=np.array(data1).reshape(len(data1),1)

df =  pd.read_table(r"E:\python\min.txt")
# print(df)
df = pd.DataFrame(df)
# print(df)
# print(df.loc[0])
#print(df.loc[1])

y2 = df['幅值 - 曲线 0']
data2 = np.array(y2)

data2=np.array(data2).reshape(len(data2),1)


def ber(data, windowlen, buchang):
    p = len(data)
    L = list()
    q = (p - windowlen)
    for i in range(q):
        if int(i) % int(buchang) == 0:
            a = data[i:i + windowlen]
            L.append(a)
    return L


a = ber(data1, 512, 6)
L1 = np.array(a)
x, y, z = L1.shape
print(L1.shape)
c = L1.reshape(x, y)
k = np.zeros((x, 1))
# print(k.shape)
KK = np.hstack([c, k])
#print(c.shape)
# print(KK.shape)

# print(data1.shape)
data2=data2[0:3072]
# print(data2.shape)
b = ber(data2, 512, 6)
L2 = np.array(b)
XX, YY, ZZ = L2.shape
# print(L2.shape)
C = L2.reshape(XX, YY)
K = np.zeros((XX, 1)) + 1
# print(K.shape)
kk = np.hstack([C, K])
#print(K)
# print(kk.shape)


L3 = np.vstack([KK, kk])
X = np.expand_dims(L3[:, 0:512].astype(float), axis=2)  # 每行的1-1024列作为训练数据
Y = L3[:, 512]  # 每行的第1025列作为训练标签
# print(L3)
# print(X)
# # print(L3.shape)
# print(L3[:, 0:512].astype(float))




# 把Y编辑成独热码,把标签做成二进制的独热码，有这一段,即使标签是汉字,仍然可以进行训练,否则不行
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)

# print(Y_encoded)
# print(Y_onehot)
#

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)
# 这一行的意思是,从读入的数据集中拿出30%作为测试集,剩下70%作为训练集。







# 卷积过程定义
model = Sequential()  # 使用序列函数，让数据按照序列排队输入到卷积层
model.add(Convolution1D(16, 16, strides=4, padding='same', input_shape=(512, 1), activation='tanh'))  # 第一个卷积层
model.add(Dropout(0.5))  # 将经过第一个卷积层后的输出数据按照0.5的概率随机置零，也可以说是灭活
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
# 添加批量标准层，将经过dropout的数据形成正太分布，有利于特征集中凸显，里面参数不需要了解和改动，直接黏贴或者删去均可。
model.add(MaxPooling1D(2, strides=2, padding='same'))
# 添加池化层，池化核大小为2步长为2，padding数据尾部补零。池化层不需要设置通道数，但卷积层需要。
model.add(Convolution1D(32, 3, padding='same', activation='tanh'))  # 第二个卷积层，第二个卷积层则不在需要设置输入数据情况。
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
# SeparableConv1D
model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第三个卷积层
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling1D(2, strides=2, padding='same'))
model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第四个卷积层
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling1D(2, strides=2, padding='same'))
model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第五个卷积层
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling1D(2, strides=2, padding='same'))
model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第六个卷积层
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                             beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling1D(2, strides=2, padding='same'))
model.add(Flatten())  # 将经过卷积和池化的数据展平，具体操作方式可以理解为，有n个通道的卷积输出，将每个通道压缩成一个数据，这样展评后就会出现n个数据
model.add(Dense(100))
model.add(Dense(2, activation='softmax', name='b'))  # 最后一层的参数设置要和标签种类一致，而且激活函数采取分类器softmax
print(model.summary())  # 模型小结，在训练时可以看到网络的结构参数
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 定义训练模型时的loss函数，acc函数，以及优化器，这个adam优化器不用导入，属于自适应优化器，这样我在一开始导入的SGD就没用了。



from keras.callbacks import TensorBoard
import time
import keras

model_name = "模型名-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs'.format(model_name), write_images='Ture')
tensorboard = keras.callbacks.TensorBoard(histogram_freq=1)
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), batch_size=64,
                    callbacks=[tensorboard])
# 训练批量大小和批次
# 定义训练的正反向传播，epochs训练批次，batch_size每个批次的大小集每次有多少个数据参与训练，数字不能太大，也不能太小，为数据集的20%为宜


import matplotlib.pyplot as plt  # 这再次导入plt画图工具

plt.figure(figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimSun']#这一行已经画不出东西了，留着吧
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.plot(history.history['loss'],'g--',label="Train_loss",  )#画出train_loss曲线
# plt.plot(history.history['val_loss'],'k:',label='Test_loss', )#画出val_loss曲线
# plt.plot(history.history['accuracy'],'b-.',label='Train_acc',  )#画出train_acc曲线
# plt.plot(history.history['val_accuracy'],'r-',label='Test_acc', )#画出val_acc曲线
plt.plot(history.history['loss'], label="Train_loss", )  # 画出train_loss曲线
plt.plot(history.history['val_loss'], label='Test_loss', )  # 画出val_loss曲线
plt.plot(history.history['accuracy'], label='Train_acc', )  # 画出train_acc曲线
plt.plot(history.history['val_accuracy'], label='Test_acc', )  # 画
plt.legend()
plt.show()  # 显示画出的曲线图