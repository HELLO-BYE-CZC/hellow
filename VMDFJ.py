# *_*coding:utf-8 *_*
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft,ifft

# f = pd.read_csv("E:\python\min_max_all\max.txt", sep="\t")
# f = f.iloc[:, -1]
# x = pd.read_csv("E:\python\min_max_all\max.txt", sep="\t")
# x = x.iloc[:,0]
# print(x)
# f = pd.read_csv('E:\python\min_max_all\max.txt')
# f = np.array(f)
# x = np.linspace(0,len(f),len(f))
f = pd.read_csv(r"E:\python\min_max_all\max.txt", sep="\t")
f = f["幅值 - 曲线 0"]
i = pd.read_csv(r"E:\python\min_max_all\min.txt", sep="\t")
i = i["幅值 - 曲线 0"]
x = np.linspace(0, len(f), len(f))

alpha = 2000       # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K =4              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7


#. Run VMD
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)


#. Visualize decomposed modes
plt.figure()
plt.subplot(2,1,1)
plt.plot(f)
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(u.T)
# plt.plot(u_hat)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.tight_layout()
print(omega[-1])

print(u.shape)

plt.figure(figsize=(6,6))
plt.subplot(1 + u.shape[0], 1, 1 )
plt.plot(f)
for i in range(u.shape[0]):

    plt.subplot(1 + u.shape[0],1,2+i)
    plt.plot(x,u[i,:],'b')
    plt.title("VMD"+str(i))
plt.tight_layout()
plt.show()



f = np.array(f)
Fs = 100.0     # sampling rate采样率
Ts = 1.0/Fs    # sampling interval 采样区间
t = np.arange(len(f))  # time vector,这里Ts也是步长

n = len(f)     # length of the signal
print(n)
k = np.arange(n)
T = n/Fs
frq = k/T     # two sides frequency range
frq1 = frq[range(int(n/2))] # one side frequency range

YY = fft(f)
N = len(YY)
half_x = np.arange(int(N / 2)) # 取一半区间
abs_y = np.abs(YY)



plt.figure(figsize=(6,6))
plt.subplot(1 + u.shape[0], 1, 1 )
plt.plot(frq1,abs_y[range(int(N / 2))])
for i in range(u.shape[0]):
    plt.subplot(1 + u.shape[0], 1, 2 + i)
    YY = fft(u[i, :])
#     print(u[i, :])

    N = len(YY)
#     print(N)
    half_x = np.arange(int(N / 2)) # 取一半区间
#     print(half_x)
    abs_y = np.abs(YY)  # 取复数的绝对值，即复数的模(双边频谱)
    normalization_y = abs_y / N  # 归一化处理（双边频谱）
    yy = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    plt.plot(frq1,abs_y[range(int(N / 2))], 'b')
plt.tight_layout()
plt.show()