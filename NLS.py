# *_*coding:utf-8 *_*
import os
import scipy.io
from PIL import Image
from scipy.linalg import logm
import numpy as np
import pandas as pd
from PyEMD import EMD, CEEMDAN, Visualisation, EEMD
import math
from pandas import Series
from scipy.signal import argrelextrema
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
import matplotlib.pyplot as plt


def get_data(folder):
    ''' 获取某个工况下某个轴承的全部n个csv文件中的数据，返回numpy数组
    dp:bearings_x_x的folder
    return:folder下n个csv文件中的数据，shape:[n*32768,2]=[文件个数*采样点数，通道数]'''
    names = os.listdir(folder)

    is_acc = ['2021' in name for name in names]
    names = names[:sum(is_acc)]
    # print(names)
    files = [os.path.join(folder, f) for f in names]
    # print(files)
    L = []
    for j in range(len(files)):

        print(files[j], str(files[j]))
        a = pd.read_csv(files[j])
        a = a[11:3072]

        def ber(data, windowlen, buchang):
            p = len(data)

            q = (p - windowlen)
            for i in range(q):
                if int(i) % int(buchang) == 0:
                    a = data[i:i + windowlen]

                    np.savetxt('E:\python\\nengliang\chuli\\max1\\' +"2021" +  str(j) +  str(i//buchang)+'.txt', a)
                    L.append(a)
                    print(L)
            return L

        b = ber(a, 512, 256)
    #     np.savetxt(r'E:\python\nengliang\chuli\''+ str(i) +'.txt',b)
    #     #     B= np.array(B)
    #     #     print(B,len(B),B.shape)
    return b


# B = get_data('E:\python\min_max_all\min-max\max')



#能量熵
def nengliangshang(folder):
    H = []
    names = os.listdir(folder)

    # is_acc = ['2021' in name for name in names]
    # names = names[:sum(is_acc)]
    # print(names)
    files = [os.path.join(folder, f) for f in names]
    print(files)
    for j in range(len(files)):
        a = np.loadtxt(files[j])
        eemd = EEMD(trials=10)
        imfs_eemd = eemd(a)
        Ei = []
        for s in range(4):
            a = imfs_eemd[s, :]
            # print(len(a))
            E = a ** 2
            E = sum(E)
            Ei.append(E)
        # print(Ei)
        E = sum(Ei)
        # print(E)
        pi = Ei / E
        # print(pi,type(pi))
        # print(np.log2(pi),np.log10(pi))
        # print(pi[0],np.log2(pi[0]),np.log10(pi[0]))
        # print(pi * np.log10(pi))
        h = -sum(pi * np.log2(pi))
        H.append(h)
        print('正在打印第%d个' % j)
        np.savetxt('E:\\python\\nengliang\\chuli\\max\\max1nengliangshang.txt',H)

# nengliangshang(r"E:\python\nengliang\chuli\max1")

#画图
def huatu(path1,path2):
    a = np.loadtxt(path1)
    b = np.loadtxt(path2)
    print(a)
    x = np.linspace(0 ,len(a),len(a))
    x = np.array(x)
    print(x, x.shape)

    plt.figure()
    # plt.plot(a,'b',label="max")
    # plt.plot(b, 'c',label="min")
    plt.scatter(x, a)
    plt.scatter(x, b)
    # plt.legend()
    plt.show()

# huatu('E:\\python\\nengliang\\chuli\\max1\\max1nengliangshang.txt','E:\\python\\nengliang\\chuli\\min1\\min1nengliangshang.txt')
#


#svm

def fenlei(path1,path2):
    a = np.loadtxt(path1)
    b = np.loadtxt(path2)
    x = [a,b]
    x = np.array(x)
    x = x.reshape(100,1)

    g = np.linspace(0,40,40)
    print(g)
    g = np.array(g)
    print(g)
    g = g.reshape(40,1)
    # print(x)

    # x = np.array(x)
    # c = []
    #
    # for i in range(50):
    #     c.append(0)
    # c = np.array(c)
    # d= []
    # for i in range(50):
    #     d.append(1)
    # d = np.array(d)
    # y = [c,d]
    # y = np.array(y)
    # y = y.reshape(100,1)
    # np.savetxt('E:\\python\\nengliang\\chuli\\ybiaoqian.txt',y)
    y = np.loadtxt('E:\\python\\nengliang\\chuli\\ybiaoqian.txt')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)  # 分割数据集
    clf = svm.SVC(C = 1.0, kernel='rbf')  # SVM模块，svc,线性核函数
    clf.fit(X_train, y_train)
    # thresholds = np.linspace(2, 3, 900)
    # # Set the parameters by cross-validation
    # param_grid = {'gamma': thresholds}
    #
    # clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
    # clf.fit(X_train, y_train)
    # print("best param: {0}\nbest score: {1}".format(clf.best_params_,
    #                                                 clf.best_score_))

    y_hat =clf.predict(X_test)
    print('准确率',accuracy_score(y_test,y_hat))
    h = clf.support_vectors_
    plt.figure(figsize=(12,8))
    plt.scatter(g,X_test)
    plt.plot(h)
    plt.show()

# fenlei('E:\\python\\nengliang\\chuli\\max1\\max1nengliangshang.txt','E:\\python\\nengliang\\chuli\\min1\\min1nengliangshang.txt')


# def qiaodu(folder):
#     qiaodu = []
#     names = os.listdir(folder)
#
#     files = [os.path.join(folder, f) for f in names]
#     print(files)
#     for j in range(len(files)):
#         a = pd.read_csv(files[j],header=None)
#         a = a.kurt()
#         qiaodu.append(a)
#     print(type(a))
#
#     plt.figure(figsize=(5, 5))
#     plt.plot(qiaodu)
#     plt.title('qiaodu')
#     plt.show()
#     return qiaodu
#     # np.savetxt(path + 'qiaodo.txt', qiaodu)
# qq = qiaodu(r"E:\python\nengliang\chuli\max")
# qqq = qiaodu(r"E:\python\nengliang\chuli\min")
# plt.figure(figsize=(5, 5))
# plt.plot(qq)
# plt.plot(qqq)
# plt.title('qiaodu')
# plt.show()


a = pd.read_csv(r"E:\python\min_max_all\max.txt", sep="\t")
a = a["幅值 - 曲线 0"]
i = pd.read_csv(r"E:\python\min_max_all\min.txt", sep="\t")
i = i["幅值 - 曲线 0"]
x = np.linspace(0, len(a), len(a))

i = i[0:3072]


