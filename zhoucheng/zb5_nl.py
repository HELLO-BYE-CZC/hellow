# *_*coding:utf-8 *_*
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#使用sklearn库中自带的iris数据集作为示例
x = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\zb6.txt',header=None)
x = np.array(x)
print(x.shape)
y = pd.read_csv(r'E:\Download\zhenghaiyang-phm-ieee-2012-data-challenge-dataset-master\phm-ieee-2012-data-challenge-dataset\txt\PCAguiyihua.txt',header=None)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4) #分割数据集
# print(X_train, X_test, y_train, y_test)

# clf = svm.SVR(kernel='rbf',C=1.47605502e+00,gamma=1.44443597e-03,epsilon=1.60926200e-01)            #shuff=true  100粒子 x[0:650:4] r2    jun    预测730


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