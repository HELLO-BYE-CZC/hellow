# *_*coding:utf-8 *_*
# *_*coding:utf-8 *_*
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#使用sklearn库中自带的iris数据集作为示例
x = pd.read_csv(r'E:\Download\pso-svm-master\pso-svm-master\data\x.txt',header=None)
x = np.array(x)
x = x.reshape(1085,1)
# x = x.reshape(-1, 1)
y = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\nengliangshangguiyihua.txt',header=None)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4) #分割数据集
# X_train, X_test, y_train, y_test = x[0:650],x[650:],y[0:650],y[650:]
# print(X_train, X_test, y_train, y_test)
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]

# print(x[730:830].values)
# clf = svm.SVR(kernel='rbf',C=8.16541625,gamma=0.24957729,epsilon=0.10872141)        #1000粒子 1000次  nengliangshangguiyihua
# clf = svm.SVR(kernel='rbf',C=3.44822729,gamma=0.5552085,epsilon=0.0546856)            #shuff=true  1000粒子  r2 0.63
# clf = svm.SVR(kernel='rbf',C=1.47605502e+00,gamma=1.44443597e-03,epsilon=1.60926200e-01)            #shuff=true  100粒子 x[0:650:6] r2 0.87   jun 0.015   预测730
clf = svm.SVR(kernel='rbf',C=6.5701406,gamma=0.1313934 ,epsilon=0.10636511)            #shuff=true  1000粒子  r2 0.79


# clf.fit(X_train[0:650:4], y_train[0:650:4])
# y_hat = clf.predict(x[730:830].values)
# print(y_hat)
# print("得分:", r2_score(y[730:830], y_hat))
# print('均方差',mean_squared_error(y[730:830], y_hat))
# # print(X_test,y_hat)
# plt.plot(x[730:830].values,y_hat, 'go-', label="predict")
# plt.plot(x[730:830].values,y[730:830].values, 'co-', label="real")
# plt.legend()
# plt.show()

#
# y_hat = clf.predict(x)
# print(y_hat[130:230])
#
# # print("得分:", r2_score(y_test, y_hat))
#
# print(X_test,y_hat)
# plt.plot(x[130:230].values,y_hat[130:230], 'go-', label="predict")
# plt.plot(x[130:230].values,y[130:230].values, 'co-', label="real")
# plt.legend()
# plt.show()


# clf.fit(X_train, y_train)
# y_test_pred = clf.predict(x)
# plt.plot(y)
# plt.plot(y_test_pred)
# plt.show()
# print("得分:", r2_score(y, y_test_pred))

# clf.fit(X_train,y_train)
# y_test_pred = clf.predict(x)
# plt.plot(y)
# plt.plot(y_test_pred)
# plt.show()

clf.fit(X_train, y_train)
y_hat = clf.predict(x[730:830,:])
y_hat1 = clf.predict(X_test)
# print(y_hat)
print("得分:", r2_score(y_test, y_hat1))
print("得分730:", r2_score(y[730:830], y_hat))
print('均方差',mean_squared_error(y[730:830], y_hat))
# print(X_test,y_hat)
plt.plot(x[730:830],y_hat, 'r^-', label="predict",markerfacecolor='white')
plt.plot(y[730:830], 'co-', label="real",markerfacecolor='white')
plt.legend()
plt.show()

for i in range(2,8):
    # clf.fit(x[0:i*100+30],y[0:i*100+30])
    y_hat = clf.predict(x[i*100+30:i*100+130])
    print(i)
    print("r2_score:", r2_score(y[i*100+30:i*100+130], y_hat))
    print('mean_squared_error',mean_squared_error(y[i*100+30:i*100+130], y_hat))