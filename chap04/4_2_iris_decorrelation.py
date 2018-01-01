# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# %%
iris = datasets.load_iris()
data = iris.data
print(iris.feature_names)
print(data)

# %%
cov = np.cov(data[:, 2:4], rowvar=0)
v, S = np.linalg.eig(cov)
S

# %%
data_dec = np.dot(S.T, data[:, 2:4].T).T

# %%
plt.figure(figsize=(6, 6))
plt.scatter(data[0: 50, 2], data[0: 50, 3], marker='$s$')
plt.scatter(data[50:100, 2], data[50:100, 3], marker='$c$')
plt.scatter(data[100:150, 2], data[100:150, 3], marker='$v$')
plt.xlim(1, 7)
plt.ylim(-2, 4)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Before decorrelation')
plt.show()
plt.savefig('./chap04/4_2_iris_decorrelation_before.eps')

# %%
plt.figure(figsize=(6, 6))
plt.scatter(data_dec[0: 50, 0], data_dec[0: 50, 1], marker='$s$')
plt.scatter(data_dec[50:100, 0], data_dec[50:100, 1], marker='$c$')
plt.scatter(data_dec[100:150, 0], data_dec[100:150, 1], marker='$v$')
plt.xlim(1, 7)
plt.ylim(-3, 3)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('After decorrelation')
plt.show()
plt.savefig('./chap04/4_2_iris_decorrelation_after.eps')
