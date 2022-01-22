# *_*coding:utf-8 *_*
import  matplotlib.pyplot as plt
import numpy as np
import  pandas as pd

#5个最大最小信号 窗长512 步长256 分解50个 能量熵

def fj50():
    p = np.loadtxt('E:/python/nengliang/chuli/max/maxnengliangshang.txt')
    d = np.loadtxt('E:/python/nengliang/chuli/min/minnengliangshang.txt')
    p1 = np.loadtxt('E:/python/nengliang/chuli/max1/max1nengliangshang.txt')
    p11 = np.loadtxt('E:/python/nengliang/chuli/max1/max1vmdnengliangshang.txt')

    d1 = np.loadtxt('E:/python/nengliang/chuli/min1/min1nengliangshang.txt')
    d11 = np.loadtxt('E:/python/nengliang/chuli/min1/min1vmdnengliangshang.txt')

    x = np.linspace(0, len(p), len(p))
    plt.figure(figsize=(22,18))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(221)
    plt.title('eemd能量熵乱顺序散点')
    plt.scatter(x,p,label="max")
    plt.scatter(x,d,label="min")
    plt.legend()
    plt.subplot(222)
    plt.title('eemd能量熵乱顺序')
    plt.plot(p)
    plt.plot(d)
    plt.subplot(223)
    plt.title('vmd能量熵按顺序')
    plt.plot(p11)
    plt.plot(d11)
    plt.subplot(224)
    plt.title('eemd能量熵按顺序')
    plt.plot(p1)
    plt.plot(d1)
    plt.show()

# fj50()



#max min 512 长度 等分6份
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

huatu('E:\\python\\min_max_all\\max\maxdf\\max1vmdnengliangshang.txt','E:\\python\\min_max_all\\min\mindf\\min1vmdnengliangshang.txt')




#