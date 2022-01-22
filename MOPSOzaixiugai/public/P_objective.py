import numpy as np
from vmdpy import VMD
import pandas as pd
import scipy.signal as signal

def P_objective(Operation,Problem,M,Input):
    [Output, Boundary, Coding] = P_DTLZ(Operation, Problem, M, Input)
    if Boundary == []:
        return Output
    else:
        return Output, Boundary, Coding

def P_DTLZ(Operation,Problem,M,Input):
    Boundary = []
    Coding = ""
    FunctionValue = []
    k = 1
    K = [5, 10, 10, 10, 10, 10, 20]
    K_select = K[k - 1]
    if Operation == "init":
        D = 2
        MaxValue = np.ones((1, D))*np.inf
        MinValue = np.ones((1, D))*-100
        x = []
        for i in range(1, 10, 2):
            for j in range(500, 3000, 10):
                x.append([i, j])
        x = np.array(x)

        Population = x

        Boundary = np.vstack((MaxValue, MinValue))
        Coding = "Real"
        return Population, Boundary, Coding
    elif Operation == "value":



        if Problem == "DTLZ1":

            pass

        elif Problem =="VMD":

            Population = Input
            d = pd.read_csv('E:\\CAET\\有车\\1-1-2021-12-15-16-34-10.txt')[10:]
            d = np.array(d).reshape(1, -1)[0]

            FunctionValue = []
            for i in range(len(Population)):
                b = []
                Rk = []

                alpha = Population[i, :][1]  # moderate bandwidth constraint
                tau = 0.  # noise-tolerance (no strict fidelity enforcement)
                K = Population[i, :][0]  # 3 modes
                if K <= 0 :
                    K =1
                DC = 0  # no DC part imposed
                init = 1  # initialize omegas uniformly
                tol = 1e-7
                # . Run VMD
                u, u_hat, omega = VMD(d, alpha, tau, K, DC, init, tol)

                def bls(y, fs):

                    y_hht = signal.hilbert(y)  # ;%希尔伯特变换
                    y_an = abs(y_hht)  # ;%包络信号
                    y_an = y_an - np.mean(y_an)  # ;%去除直流分量
                    y_an_nfft = np.log2(len(y_an))  # ;%包络的DFT的采样点数 取2的幂次方
                    #     print(y_an_nfft,y_an)
                    y_an_ft = np.fft.fft(y_an, int(y_an_nfft))  # ;%包络的DFT

                    c = 2 * abs(y_an_ft[0:int(y_an_nfft / 2)]) / len(y_an)  # ;%包络幅值

                    p = c / np.sum(c)
                    E = -sum(p * np.log(p))
                    return E

                for j in range(K):
                    R = u[j:]


                    R_mean = np.mean(R)  # 计算均值
                    R_var = np.var(R)  # 计算方差
                    R_ku = np.mean((R - R_mean) ** 4) / pow(R_var, 2)  # 计算峰度
                    Rk.append(-R_ku)

                    fs = 100
                    bl = bls(u[j], fs)
                    b.append(bl)

                    # print('*****************************************************')
                B = np.array(b).reshape(-1, 1)
                RK = np.array(Rk).reshape(-1, 1)
                # print(b, Rk)
                f = np.hstack((B, RK))
                ind = np.lexsort((b, Rk)).tolist()
                # print(ind,type(ind))
                ind = ind.index(max(ind))
                # print('suoyin',ind)
                # print("ind_max%d"%(np.max(ind)))
                # print(f,f[ind])
                f = f[ind]

                FunctionValue.append(f)
            FunctionValue =np.array(FunctionValue)
    return FunctionValue, Boundary, Coding








