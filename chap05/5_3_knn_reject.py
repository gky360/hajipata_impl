# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K_MAX = 60
C = 2

# %% Load Pima dataset

p_df_tr = pd.read_csv(os.path.join("data", "Pima.tr.csv"))
p_df_te = pd.read_csv(os.path.join("data", "Pima.te.csv"))
p_df = p_df_tr.append(p_df_te, ignore_index=True)

scaler = StandardScaler()
scaler.fit(p_df_tr[['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']])
p_x = scaler.transform(
    p_df[['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']])
p_y = np.array([1 if t == 'Yes' else 0 for t in p_df['type']])

p_x_tr = p_x[0:len(p_df_tr)]
p_y_tr = p_y[0:len(p_df_tr):]
p_x_te = p_x[len(p_df_tr):]
p_y_te = p_y[len(p_df_tr):]

# %% Prepare KNN classifier


def predict_knn(knn, x_tr, y_tr, x_te, y_te, n_classes=2):
    knn.fit(x_tr, y_tr)

    def vote_with_reject(arr, n_classes=n_classes):
        cnt = np.bincount(arr, minlength=n_classes)
        c = cnt.argmax()
        c_rev = n_classes - 1 - cnt[::-1].argmax()
        if c != c_rev:
            # reject
            return n_classes
        return c

    neigh_idx = knn.kneighbors(x_te, return_distance=False)
    pred = np.apply_along_axis(lambda x: vote_with_reject(
        x, n_classes=n_classes), axis=1, arr=y_tr[neigh_idx])
    pred_idx = pred < n_classes
    score = np.count_nonzero(
        pred[pred_idx] == y_te[pred_idx]) / np.count_nonzero(pred_idx)
    return 1 - score, (len(x_te) - np.count_nonzero(pred_idx)) / len(x_te)


def knn_holdout(x_tr, y_tr, x_te, y_te, max_k):
    p_error = np.zeros(max_k)
    p_reject = np.zeros(max_k)
    for i in range(max_k):
        k = i + 1
        knn = KNeighborsClassifier(n_neighbors=k)
        e, r = predict_knn(knn, x_tr, y_tr, x_te, y_te, n_classes=C)
        p_error[i] = e
        p_reject[i] = r

    return p_error, p_reject


# %% Predict

p_error, p_reject = knn_holdout(p_x_tr, p_y_tr, p_x_te, p_y_te, K_MAX)

# %% Plot results

plt.figure(figsize=(6, 6))
plt.scatter(range(1, K_MAX + 1, 2), p_error[::2],
            marker='x', label='odd')
plt.scatter(range(2, K_MAX + 1, 2), p_error[1::2],
            marker='o', label='even')
plt.xlabel('k')
plt.ylabel('Error rate')
plt.title('Erro rate')
plt.legend()
plt.savefig('./chap05/5_3_knn_reject_error_rate.png')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(range(2, K_MAX + 1, 2), p_reject[1::2],
            marker='o')
plt.xlabel('k')
plt.ylabel('Reject rate')
plt.title('Reject rate')
plt.savefig('./chap05/5_3_knn_reject_reject_rate.png')
plt.show()
