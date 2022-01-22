#encoding: utf-8
import random
import numpy as np
from public import pareto,NDsort


def init_designparams(particals,in_min,in_max):
    in_dim = len(in_max)     #输入参数维度
    # print(len(in_max))
    '''in_temp = np.random.uniform(0,1,(6,in_dim))*(in_max-in_min)+in_min'''
    q = []
    for i in range(particals):
        k = random.randint(3, 12)
        a = random.randint(500, 4000)
        s = [k, a]
        q.append(s)
        print(s)
    in_temp =np.array(q)
    print("in_temp")
    print(in_temp)
    # print(np.random.uniform(0,1,(particals,in_dim)),(in_max-in_min)+in_min,np.random.uniform(0,1,(particals,in_dim))*(in_max-in_min))
    return in_temp

def init_v(particals,v_max,v_min):
    v_dim = len(v_max)     #输入参数维度
    # v_ = np.random.uniform(0,1,(particals,v_dim))*(v_max-v_min)+v_min

    v_ = np.zeros((particals,v_dim))
    print('v_',v_)
    return v_

def init_pbest(in_,fitness_):
    return in_,fitness_

def init_archive(in_,fitness_):
    print('init_archive',in_)
    FrontValue_1_index = NDsort.NDSort(fitness_, in_.shape[0])[0]==1
    # print(FrontValue_1_index)
    FrontValue_1_index = np.reshape(FrontValue_1_index,(-1,))

    curr_archiving_in=in_[FrontValue_1_index]
    curr_archiving_fit=fitness_[FrontValue_1_index]

    # pareto_c = pareto.Pareto_(in_,fitness_)
    # curr_archiving_in_,curr_archiving_fit_ = pareto_c.pareto()
    return curr_archiving_in,curr_archiving_fit


