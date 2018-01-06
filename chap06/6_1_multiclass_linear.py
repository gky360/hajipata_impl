# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import datasets

# %% Load iris dataset
iris = datasets.load_iris()
data = iris.data[:, 2:4]
N = len(data)

# %% Prepare matrices
X = np.append(np.ones((N, 1)), data, axis=1)
T = np.zeros((N, 3))
T[np.arange(N), iris.target] = 1

# %% Fit

W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(T)
W.shape

# %% Classification

X_MIN, X_MAX = 0.0, 7.0
X_STEP = (X_MAX - X_MIN) / 300
Y_MIN, Y_MAX = 0.0, 3.0
Y_STEP = (Y_MAX - Y_MIN) / 300

xx, yy = np.meshgrid(np.arange(X_MIN, X_MAX, X_STEP),
                     np.arange(Y_MIN, Y_MAX, Y_STEP))
Z = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()].dot(W).argmax(axis=1)
Z = Z.reshape(xx.shape)

# %% Plot

plt.pcolormesh(xx, yy, Z, alpha=0.1, cmap=cm.brg)
plt.scatter(data[iris.target == 0, 0],
            data[iris.target == 0, 1], marker='$s$', c='b')
plt.scatter(data[iris.target == 1, 0],
            data[iris.target == 1, 1], marker='$c$', c='r')
plt.scatter(data[iris.target == 2, 0],
            data[iris.target == 2, 1], marker='$v$', c='g')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Iris dataset')
plt.savefig('./chap06/6_1_multiclass_linear_iris.png')
plt.show()
