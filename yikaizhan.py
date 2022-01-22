# *_*coding:utf-8 *_*
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from PyEMD import EMD,CEEMDAN,Visualisation,EEMD





df =  pd.read_table(r"E:\python\min.txt")

df = pd.DataFrame(df)

y = df['幅值 - 曲线 0']

y = np.array(y)
ii =list(range(0,len(y),2))
print(y)
print(ii)
# y = y[ii]
# print(len(y))

plt.figure(figsize=(15,8))
plt.plot(y)
plt.show()


from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号



Fs = 50.0     # sampling rate采样率
Ts = 1.0/Fs    # sampling interval 采样区间
t = np.arange(len(y))  # time vector,这里Ts也是步长

n = len(y)     # length of the signal
print(n)
k = np.arange(n)
T = n/Fs
frq = k/T     # two sides frequency range
frq1 = frq[range(int(n/2))] # one side frequency range


# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.arange(len(y))

# 设置需要采样的信号，频率分量有200，400和600

fft_y = fft(y)  # 快速傅里叶变换

N = len(y)
# x = np.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度
normalization_y = abs_y / N  # 归一化处理（双边频谱）
yy = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

plt.subplot(211)
plt.plot(x, y)
plt.title('原始波形')

'''# plt.subplot(232)
# plt.plot(x, fft_y, 'black')
# plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

# plt.subplot(233)
# plt.plot(frq, abs_y, 'r')
# plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')
#
# plt.subplot(234)
# plt.plot(x, angle_y, 'violet')
# plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')
#
# plt.subplot(235)
# plt.plot(x, normalization_y, 'g')
# plt.title('双边振幅谱(归一化)', fontsize=9, color='green')'''

plt.subplot(212)
plt.plot(frq1, yy, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
plt.tight_layout()
plt.show()





##########################################################################################

'''y = y
emd = EMD()
eemd = EEMD()
ceemdan = CEEMDAN()
imfs_emd = emd(y)
imfs_eemd = eemd(y)
imfs_ceemd = ceemdan(y)

print(np.shape(imfs_emd))


plt.figure(1,figsize=(12,10))
plt.subplot(1 + np.shape(imfs_emd)[0], 1, 1 )
plt.plot(frq1, y, 'r')
plt.title("Signal Input")
for i in range(np.shape(imfs_emd)[0]):
    plt.subplot(1 + np.shape(imfs_emd)[0],1,2+i)
    plt.plot(frq1,imfs_emd[i,:],'b')
    plt.title("IMF-emd"+str(i))
plt.tight_layout()
plt.show()


plt.figure(2,figsize=(12,10))
plt.subplot(1 + np.shape(imfs_eemd)[0], 1, 1 )
plt.plot(frq1, y, 'r')
plt.title("Signal Input")
for i in range(np.shape(imfs_eemd)[0]):
    plt.subplot(1 + np.shape(imfs_eemd)[0],1,2 + i)
    plt.plot(frq1, imfs_eemd[i, :], 'b')
    plt.title("IMF-eemd" + str(i))
plt.tight_layout()
plt.show()

plt.figure(3,figsize=(12,10))
plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 1 )
plt.plot(frq1, y, 'r')
plt.title("Signal Input")

for i in range(np.shape(imfs_ceemd)[0]):
    plt.subplot(1 + np.shape(imfs_ceemd)[0],1,2 + i)
    plt.plot(frq1, imfs_ceemd[i, :], 'b')
    plt.title("IMF-ceemdan" + str(i))
plt.tight_layout()
plt.show()
###############p=2 show
plt.figure(4,figsize=(12,10))
plt.subplot(3,2,1)
plt.plot(frq1, y, 'r')
plt.title("Signal Input")
plt.xlabel('Time /s')
plt.ylabel('Intensity')
'''
##################################################################
'''y = y
emd = EMD()
eemd = EEMD()
ceemdan = CEEMDAN()
imfs_emd = emd(yy)
imfs_eemd = eemd(yy)
imfs_ceemd = ceemdan(yy)

print(np.shape(imfs_emd))


plt.figure(1,figsize=(12,16))

plt.subplot(2 + np.shape(imfs_emd)[0], 1, 1 )
plt.plot(x, y, 'r')
plt.subplot(2 + np.shape(imfs_emd)[0], 1, 2 )
plt.plot(frq1,yy)
plt.title("Signal Input")
for i in range(np.shape(imfs_emd)[0]):
    plt.subplot(2 + np.shape(imfs_emd)[0],1,3+i)
    plt.plot(frq1,imfs_emd[i,:],'b')
    plt.title("IMF-emd"+str(i))
plt.tight_layout()
plt.show()


plt.figure(2,figsize=(12,16))
plt.subplot(2 + np.shape(imfs_eemd)[0], 1, 1 )
plt.plot(x, y, 'r')
plt.subplot(2 + np.shape(imfs_eemd)[0], 1, 2 )
plt.plot(frq1,yy)
plt.title("Signal Input")
for i in range(np.shape(imfs_eemd)[0]):
    plt.subplot(2 + np.shape(imfs_eemd)[0],1,3 + i)
    plt.plot(frq1, imfs_eemd[i, :], 'b')
    plt.title("IMF-eemd" + str(i))
plt.tight_layout()
plt.show()

plt.figure(3,figsize=(12,16))
plt.subplot(2 + np.shape(imfs_ceemd)[0], 1, 1 )
plt.plot(x, y, 'r')
plt.title("Signal Input")
plt.subplot(2 + np.shape(imfs_ceemd)[0], 1, 2 )
plt.plot(frq1,yy)
for i in range(np.shape(imfs_ceemd)[0]):
    plt.subplot(2 + np.shape(imfs_ceemd)[0],1,3 + i)
    plt.plot(frq1, imfs_ceemd[i, :], 'b')
    plt.title("IMF-ceemdan" + str(i))
plt.tight_layout()
plt.show()'''
# ###############p=2 show
# plt.figure(4,figsize=(12,10))
# plt.subplot(3,2,1)
# plt.plot(x, y, 'r')
# plt.title("Signal Input")
# plt.xlabel('Time /s')
# plt.ylabel('Intensity')
#####################################################################################

y = y
emd = EMD()
eemd = EEMD()
ceemdan = CEEMDAN()
# imfs_emd = emd(y)
# imfs_eemd = eemd(yy)
imfs_ceemd = ceemdan(y)

print(np.shape(imfs_ceemd))


plt.figure(1,figsize=(12,16))

plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 1 )
plt.plot(x, y, 'r')

plt.title("Signal Input")
for i in range(np.shape(imfs_ceemd)[0]):
    plt.subplot(1 + np.shape(imfs_ceemd)[0],1,2+i)
    plt.plot(x,imfs_ceemd[i,:],'b')
    plt.title("IMF-emd"+str(i))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,16))
plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 1 )
plt.plot(frq1,yy)
for i in range(np.shape(imfs_ceemd)[0]):
    plt.subplot(1 + np.shape(imfs_ceemd)[0], 1, 2 + i)
    YY = fft(imfs_ceemd[i, :])

    N = len(YY)
    half_x = x[range(int(N / 2))]  # 取一半区间
    abs_y = np.abs(YY)  # 取复数的绝对值，即复数的模(双边频谱)
    normalization_y = abs_y / N  # 归一化处理（双边频谱）
    yy = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    plt.plot(frq1,abs_y[range(int(N / 2))], 'b')
plt.tight_layout()
plt.show()