import numpy as np


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
        MaxValue = np.ones((1, D))
        MinValue = np.ones((1, D))
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
        x = []
        for i in range(1, 10, 2):
            for j in range(500, 3000, 10):
                x.append([i, j])
        x = np.array(x)

        Population = x
        # Population = Input
        # FunctionValue = np.zeros((Population.shape[0], 2))
        if Problem == "DTLZ1":
            # g = 100*(K_select+np.sum( (Population[:, M-1:] - 0.5)**2 - np.cos(20*np.pi*(Population[:, M-1:] - 0.5)), axis=1, keepdims = True))
            g = 100*(K_select+np.sum( (Population[:, M-1:] - 0.5)**2 - np.cos(20*np.pi*(Population[:, M-1:] - 0.5)), axis=1))
            for i in range(M):
                FunctionValue[:, i] = 0.5*np.multiply( np.prod(Population[:, :M-i-1], axis=1), (1+g))
                if i>0:
                    FunctionValue[:, i] = np.multiply(FunctionValue[:, i], 1-Population[:, M-i-1])
        elif Problem == "DTLZ2":
            g = np.sum( (Population[:, M-1:] - 0.5)**2,axis=1)
            for i in range(M):
                FunctionValue[:, i] = (1+g)*np.prod( np.cos( 0.5*np.pi*(Population[:, :M-i-1]) ),axis=1 )
                if i>0:
                    FunctionValue[:, i] = np.multiply(FunctionValue[:, i], np.sin( 0.5*np.pi* ( Population[:, M-i-1]) ) )


        elif Operation =="VMD":

            # FunctionValue = []

            d = pd.read_csv('E:\\CAET\\有车\\1-1-2021-12-15-16-34-10.txt')[10:]
            d = np.array(d).reshape(1, -1)[0]

            for i in range(len(Population)):

                alpha = Population[i,:][1]   # moderate bandwidth constraint
                tau = 0.  # noise-tolerance (no strict fidelity enforcement)
                K = Population[i,:][0]  # 3 modes
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
                    #     print(len(y_an))
                    c = 2 * abs(y_an_ft[0:int(y_an_nfft / 2)]) / len(y_an)  # ;%包络幅值
                    #     print(c)
                    p = c / np.sum(c)
                    E = -sum(p * np.log(p))
                    return E

                # bls(d,fs)

                # q = d.kurt()
                # print(u.shape)
                b = []
                Rk = []
                for i in range(K):
                    R = u[i:]
                    #     print(R)

                    R_mean = np.mean(R)  # 计算均值
                    R_var = np.var(R)  # 计算方差
                    R_ku = np.mean((R - R_mean) ** 4) / pow(R_var, 2)  # 计算峰度
                    Rk.append(-R_ku)
                    #     R_ku.tolist()
                    #     print('峭度是%f'%R_ku,type(R_ku))
                    fs = 100
                    bl = bls(u[i], fs)
                    b.append(bl)
                aaa = min(b),min(Rk)
                FunctionValue.append(aaa)
    FunctionValue =np.array(FunctionValue)

    return FunctionValue, Boundary, Coding


def NDSort(PopObj,Remain_Num):

    N,M = PopObj.shape
    FrontNO = np.inf*np.ones((1, N))
    MaxFNO = 0
    PopObj, rank = sortrows.sortrows(PopObj)


    while (np.sum(FrontNO < np.inf) < Remain_Num) :
        MaxFNO += 1
        for i in range(N):
            if FrontNO[0, i] == np.inf:
                Dominated = False
                for j in range(i-1, -1, -1):
                    if FrontNO[0, j] == MaxFNO:
                        m=2
                        while (m <= M) and (PopObj[i, m-1] >= PopObj[j, m-1]):
                            m += 1
                        Dominated = m > M
                        if Dominated or (M == 2):
                            break
                if not Dominated:
                    FrontNO[0,i] = MaxFNO
    # temp=np.loadtxt("temp.txt")
    # print((FrontNO==temp).all())
    front_temp = np.zeros((1,N))
    front_temp[0, rank] = FrontNO
    # FrontNO[0, rank] = FrontNO 不能这么操作，因为 FrontNO值 在 发生改变 ，会影响后续的结果


    return front_temp, MaxFNO





def update_archive_1(in_,fitness_,archive_in,archive_fitness,thresh,mesh_div):
    ##首先，计算当前粒子群的pareto边界，将边界粒子加入到存档archiving中
    total_Pop = np.vstack((archive_in,in_))
    total_Func = np.vstack((archive_fitness,fitness_))

    FrontValue_1_index = NDsort.NDSort(total_Func, total_Pop.shape[0])[0]==1
    FrontValue_1_index = np.reshape(FrontValue_1_index,(-1,))
    archive_in =total_Pop[FrontValue_1_index]
    archive_fitness = total_Func[FrontValue_1_index]

    # if archive_in.shape[0] > thresh:
    #
    #     Del_index = Delete(archive_fitness,archive_in.shape[0]-thresh,mesh_div)
    #     archive_in  = np.delete(archive_in,Del_index,0)
    #     archive_fitness = np.delete(archive_fitness,Del_index,0)
    return archive_in,archive_fitness

class Mopso:
    def __init__(self,particals,max_,min_,thresh,mesh_div=10):


        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        # self.max_v = (max_-min_)*0.5  #速度上限
        # self.min_v = (max_-min_)*(-1)*0.5 #速度下限

        self.max_v = 100 * np.ones(len(max_), )  # 速度上限,速度不存在上下限，因此设置很大
        self.min_v = -100 * np.ones(len(min_), )  # 速度下限

        # self.plot_ = plot.Plot_pareto()

    def evaluation_fitness(self):
        self.fitness_ = P_objective("value", "VMD", 2, self.in_)

    def NDSort(PopObj, Remain_Num):

        N, M = PopObj.shape
        FrontNO = np.inf * np.ones((1, N))
        MaxFNO = 0
        PopObj, rank = sortrows.sortrows(PopObj)

        while (np.sum(FrontNO < np.inf) < Remain_Num):
            MaxFNO += 1
            for i in range(N):
                if FrontNO[0, i] == np.inf:
                    Dominated = False
                    for j in range(i - 1, -1, -1):
                        if FrontNO[0, j] == MaxFNO:
                            m = 2
                            while (m <= M) and (PopObj[i, m - 1] >= PopObj[j, m - 1]):
                                m += 1
                            Dominated = m > M
                            if Dominated or (M == 2):
                                break
                    if not Dominated:
                        FrontNO[0, i] = MaxFNO
        # temp=np.loadtxt("temp.txt")
        # print((FrontNO==temp).all())
        front_temp = np.zeros((1, N))
        front_temp[0, rank] = FrontNO
        # FrontNO[0, rank] = FrontNO 不能这么操作，因为 FrontNO值 在 发生改变 ，会影响后续的结果

        return front_temp, MaxFNO



    def initialize(self):

        #初始化粒子位置
        self.in_ = init_designparams(self.particals,self.min_,self.max_)
        #初始化粒子速度
        self.v_ = init_v(self.particals,self.max_v,self.min_v)
        #计算适应度ֵ
        self.evaluation_fitness()
        #初始化个体最优
        self.in_p,self.fitness_p = init_pbest(self.in_,self.fitness_)
        #初始化外部存档
        self.archive_in,self.archive_fitness = init_archive(self.in_,self.fitness_)
        #初始化全局最优
        self.in_g,self.fitness_g = update.update_gbest_1(self.archive_in,self.archive_fitness,self.mesh_div,self.particals)
    def update_(self):

        #更新粒子速度、位置、适应度、个体最优、外部存档、全局最优
        self.v_ = update.update_v(self.v_,self.min_v,self.max_v,self.in_,self.in_p,self.in_g)
        self.in_ = update.update_in(self.in_,self.v_,self.min_,self.max_)

        self.evaluation_fitness()

        self.in_p,self.fitness_p = update.update_pbest(self.in_,self.fitness_,self.in_p,self.fitness_p)

        self.archive_in, self.archive_fitness = update.update_archive_1(self.in_, self.fitness_, self.archive_in,
                                                                      self.archive_fitness,
                                                                      self.thresh, self.mesh_div)


        self.in_g,self.fitness_g = update.update_gbest_1(self.archive_in,self.archive_fitness,self.mesh_div,self.particals)

    def done(self,cycle_):
        self.initialize()
        # self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,-1)
        # since = time.time()
        for i in range(cycle_):
            self.update_()

            print('第',i,'代已完成，time consuming: ',np.round(time.time() - since, 2), "s")

            # self.plot_.show(self.in_,self.fitness_,self.archive_in,self.archive_fitness,i)
        return self.archive_in,self.archive_fitness



particals = 1250 #粒子群的数量
cycle_ = 3 #迭代次数
mesh_div = 10 #网格等分数量
thresh = 300#外部存档阀值



Problem = "DTLZ1"
M = 2
Population, Boundary, Coding = P_objective("init", Problem, M, particals)
print(Boundary,Population.shape, Coding)
max_ = Boundary[0]
min_ = Boundary[1]



mopso_ = Mopso(particals,max_,min_,thresh,mesh_div) #粒子群实例化
pareto_in,pareto_fitness = mopso_.done(cycle_) #经过cycle_轮迭代后，pareto边界粒子