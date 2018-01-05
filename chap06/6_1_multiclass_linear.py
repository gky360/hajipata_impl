# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# %% Load iris dataset
iris = datasets.load_iris()
data = iris.data[:, 2:4]
N = len(data)

# %% Prepare matrices
X = np.append(np.ones((N, 1)), data, axis=1)
T = np.zeros((N, 3))
T[np.arange(N), np.arange(N) // 50] = 1

# %% Classification

W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(T)
W.shape
