# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


# %% Load train dataset

dataset = pd.read_csv(os.path.join("data", "Pima.tr.csv"))
x = np.array(dataset[['glu', 'bmi']])
pos = x[dataset['type'] == 'No']
neg = x[dataset['type'] == 'Yes']
data = np.append(pos, neg, axis=0)
t = np.append(np.ones(len(pos)), np.zeros(len(neg)))

# %% Load test dataset

dataset = pd.read_csv(os.path.join("data", "Pima.te.csv"))
x_test = np.array(dataset[['glu', 'bmi']])
pos_test = x_test[dataset['type'] == 'No']
neg_test = x_test[dataset['type'] == 'Yes']
data_test = np.append(pos_test, neg_test, axis=0)
t_test = np.append(np.ones(len(pos_test)), np.zeros(len(neg_test)))

# %%

mu_pos = pos.mean(axis=0)
mu_neg = neg.mean(axis=0)
cov_pos = np.cov(pos, rowvar=0)
cov_neg = np.cov(neg, rowvar=0)
cov_inv_pos = np.linalg.inv(cov_pos)
cov_inv_neg = np.linalg.inv(cov_neg)
p_pos = len(pos) / (len(pos) + len(neg))
p_neg = len(neg) / (len(pos) + len(neg))

# %% Quadratic classification function

S_quad = cov_inv_pos - cov_inv_neg
c_quad = (mu_neg.T.dot(cov_inv_neg) - mu_pos.T.dot(cov_inv_pos)).T
F_quad = mu_pos.T.dot(cov_inv_pos).dot(mu_pos) - mu_neg.T.dot(cov_inv_neg).dot(mu_neg)\
    + np.log(np.linalg.det(cov_pos) / np.linalg.det(cov_neg))\
    - 2 * np.log(p_pos / p_neg)

def f_quad(x):
    return np.einsum('ij,ik,kj->j', x.T, S_quad, x.T) + 2 * c_quad.T.dot(x.T) + F_quad

delta = 0.25
gx, gy = np.meshgrid(np.arange(50, 200, delta), np.arange(15, 50, delta))
vx = np.array([gx, gy])
vz = np.einsum('ijk,il,ljk->jk', vx, S_quad, vx) + 2 * np.einsum('i,ijk->jk', c_quad, vx) + F_quad

plt.figure(figsize=(6, 6))
plt.scatter(pos[:, 0], pos[:, 1], marker='o')
plt.scatter(neg[:, 0], neg[:, 1], marker='^')
CS = plt.contour(vx[0], vx[1], vz, np.arange(-10, 10, 2))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlim(50, 200)
plt.ylim(15, 50)
plt.xlabel('glu')
plt.ylabel('bmi')
plt.title('Quadratic classification function')
plt.show()

# %% Linear classification function

cov_pool = p_pos * cov_pos + p_neg * cov_neg
cov_inv_pool = np.linalg.inv(cov_pool)

S_linear = np.zeros((2, 2))
c_linear = (mu_neg.T.dot(cov_inv_pool) - mu_pos.T.dot(cov_inv_pool)).T
F_linear = mu_pos.T.dot(cov_inv_pool).dot(mu_pos) - mu_neg.T.dot(cov_inv_pool).dot(mu_neg)\
    - 2 * np.log(len(pos) / len(neg))

def f_linear(x):
    return np.einsum('ij,ik,kj->j', x.T, S_linear, x.T) + 2 * c_linear.T.dot(x.T) + F_linear

delta = 0.25
gx, gy = np.meshgrid(np.arange(50, 200, delta), np.arange(15, 50, delta))
vx = np.array([gx, gy])
vz = np.einsum('ijk,il,ljk->jk', vx, S_linear, vx)\
        + 2 * np.einsum('i,ijk->jk', c_linear, vx) + F_linear

plt.figure(figsize=(6, 6))
plt.scatter(pos[:, 0], pos[:, 1], marker='o')
plt.scatter(neg[:, 0], neg[:, 1], marker='^')
CS = plt.contour(vx[0], vx[1], vz, np.arange(-10, 10, 2))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlim(50, 200)
plt.ylim(15, 50)
plt.xlabel('glu')
plt.ylabel('bmi')
plt.title('Linear classification function')
plt.show()

# %% LOC curves

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_linear = sigmoid(-f_linear(data_test))
fpr_linear, tpr_linear, _ = metrics.roc_curve(t_test, y_linear, pos_label=1)
roc_auc_linear = metrics.auc(fpr_linear, tpr_linear)

y_quad = sigmoid(-f_quad(data_test))
fpr_quad, tpr_quad, _ = metrics.roc_curve(t_test, y_quad, pos_label=1)
roc_auc_quad = metrics.auc(fpr_quad, tpr_quad)

plt.figure(figsize=(6, 6))
plt.title('ROC')
plt.plot(fpr_linear, tpr_linear, 'b', label='Linear AUC = %0.2f' % roc_auc_linear)
plt.plot(fpr_quad, tpr_quad, 'b--', label='Quadratic AUC = %0.2f' % roc_auc_quad)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
