# *_*coding:utf-8 *_*
# *_*coding:utf-8 *_*
import tensorflow as tf
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import glob
from skimage import transform
from PIL import Image
import skimage
from skimage import io







path = 'E:/python/min-max/'
w =32
h =32
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
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img =io.imread(im)
            print(img.shape)
            #img = transform.resize(img, (w, h))
            print(img.shape)
            imgs.append(img)
            labels.append(idx)
        print(labels)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
print(data.shape)
N = len(label)
print(N)
# 打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]/255

label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int_(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_test = data[s:]
y_test = label[s:]


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
print(x_train.shape)
print(y_test)

class AlexNet8(Model):
    def __init__(self):
        super(AlexNet8, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        # x = self.c2(x)
        # x = self.b2(x)
        # x = self.a2(x)
        # x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


model = AlexNet8()

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'],
              )




# history = model.fit(x_train, y_train, batch_size=32, epochs=10,
#                    )
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32
                    )
#

model.evaluate(x_test,y_test,verbose=1)
model.summary()

import matplotlib.pyplot as plt  # 这再次导入plt画图工具

plt.figure(figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimSun']#这一行已经画不出东西了，留着吧
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.plot(history.history['loss'],'g--',label="Train_loss",  )#画出train_loss曲线
# plt.plot(history.history['val_loss'],'k:',label='Test_loss', )#画出val_loss曲线
# plt.plot(history.history['accuracy'],'b-.',label='Train_acc',  )#画出train_acc曲线
# plt.plot(history.history['val_accuracy'],'r-',label='Test_acc', )#画出val_acc曲线
plt.plot(history.history['loss'],'g', label="Train_loss", )  # 画出train_loss曲线
plt.plot(history.history['val_loss'],'b', label='Test_loss', )  # 画出val_loss曲线
plt.plot(history.history['accuracy'],'y', label='Train_acc', )  # 画出train_acc曲线
plt.plot(history.history['val_accuracy'],'r', label='Test_acc', )  # 画
plt.legend()
plt.show()
###############################################    show   ###############################################

