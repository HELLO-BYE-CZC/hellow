# *_*coding:utf-8 *_*
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'E:/Download/zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master/phm-ieee-2012-data-challenge-dataset/txt/'


#使用sklearn库中自带的iris数据集作为示例
x = np.loadtxt(path + 'zb6' + '.txt')
y = np.loadtxt(path + 'PCAguiyihua' + '.txt')
# x = pd.read_csv(r'E:\Download\pso-svm-master\pso-svm-master\data\x.txt',header=None)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4) #分割数据集

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


clf = svm.SVR(kernel='rbf',gamma=0.45267074,C=4.51804781,epsilon=0.01774412)            #x指标6 y归一化6 1000例子 迭代100次


# clf.fit(X_train[0:650:4], y_train[0:650:4])
# y_hat = clf.predict(x[730:830].values)
# print(y_hat)
# print("得分:", r2_score(y[730:830], y_hat))
# print('均方差',mean_squared_error(y[730:830], y_hat))
# # print(X_test,y_hat)
# plt.plot(x[730:830],y_hat, 'go-', label="predict")
# plt.plot(x[730:830],y[730:830], 'co-', label="real")
# plt.legend()
# plt.show()



# clf.fit(X_train, y_train)
# y_test_pred = clf.predict(x)
# plt.plot(y)
# plt.plot(y_test_pred)
# plt.show()
# print("得分:", r2_score(y, y_test_pred))



clf.fit(X_train, y_train)
y_hat = clf.predict(x[730:830,:])
print(y_hat)
print("得分:", r2_score(y[730:830], y_hat))
print('均方差',mean_squared_error(y[730:830], y_hat))
# print(X_test,y_hat)
plt.plot(y_hat, 'r^-', label="predict",markerfacecolor='white')
plt.plot(y[730:830], 'co-', label="real",markerfacecolor='white')
plt.legend()
plt.show()

for i in range(2,8):
    clf.fit(x[0:i*100+30],y[0:i*100+30])
    y_hat = clf.predict(x[i*100+30:i*100+130])
    print(i)
    print("r2_score:", r2_score(y[i*100+30:i*100+130], y_hat))
    print('mean_squared_error',mean_squared_error(y[i*100+30:i*100+130], y_hat))