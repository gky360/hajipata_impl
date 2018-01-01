# -*- coding: utf-8 -*-

import glob
import os
import struct

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

E_K_MAX = 20
P_K_MAX = 120

# %% Load ELT1 dataset

def normalize_image(img, out_shape):
    thr, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        x0, y0 = 0, 0
        x1, y1 = img.shape[0] + 1, img.shape[1] + 1
    else:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1

    n = np.max([x1 - x0, y1 - y0, 1])
    xm = np.clip(x0 + x1, n, 2 * img.shape[0] - n)
    ym = np.clip(y0 + y1, n, 2 * img.shape[1] - n)
    xl, xr = (xm - n) // 2, (xm + n) // 2
    yl, yr = (ym - n) // 2, (ym + n) // 2
    in_shape = img[xl:xr, yl:yr].shape

    resized_img = cv2.resize(img[xl:xr, yl:yr], out_shape)
    factors = (np.asarray(in_shape, dtype=float) / np.asarray(out_shape, dtype=float))
    sigma = np.maximum(1e-6, (factors - 1) / 2)
    blured_img = cv2.GaussianBlur(resized_img, (0, 0), sigma[0], sigmaY=sigma[1])
    blured_img -= blured_img.min()
    out_img = (blured_img * 255.0 / blured_img.max()).astype(np.uint8)
    return out_img

def convert_etl_file(filename, out_path):
    with open(filename, 'rb') as f:
        # skip = np.random.randint(0, 10000)
        # f.seek(skip * 2052)
        s = f.read(2052)
        while s:
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
            iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
            iP = iF.convert('P')
            fn = "{:04d}{:04d}{:02x}.png".format(r[0], r[2], r[3])
            # iP.save(fn, 'PNG', bits=4)
            enhancer = ImageEnhance.Brightness(iP)
            iE = enhancer.enhance(16)
            org_image = np.array(iE.getdata(), dtype=np.uint8).reshape(63, 64)
            image = normalize_image(org_image, (16, 16))
            iN = Image.fromarray(image)
            iN.save(os.path.join(out_path, fn), 'PNG')
            s = f.read(2052)

# convert_etl_file('./data/ETL1/ETL1C_01', './out/ETL1/')

# %% Load ELT1 dataset

def load_img(filename):
    scale = 1.0 / 256.0
    image = np.array(Image.open(filename), dtype=np.float32) * scale
    y = int(os.path.splitext(os.path.basename(filename))[0][-1])
    return image.reshape(1, 16 * 16), y

def load_etl1_dataset(path):
    filenames = glob.glob(os.path.join(path, '*[0-9].png'))
    load_images = np.frompyfunc(load_img, 1, 2)
    x, y = load_images(filenames)
    return np.concatenate(x), y.astype(np.uint8)

dataset = load_etl1_dataset('./out/ETL1')
idxes = np.random.choice(len(dataset[0]), 1300, replace=False)
e_x, e_y = dataset[0][idxes], dataset[1][idxes]
e_x_tr, e_x_te, e_y_tr, e_y_te = train_test_split(e_x, e_y, train_size=650, test_size=650)

# %% Load Pima dataset

p_df_tr = pd.read_csv(os.path.join("data", "Pima.tr.csv"))
p_df_te = pd.read_csv(os.path.join("data", "Pima.te.csv"))
p_df = p_df_tr.append(p_df_te, ignore_index=True)

scaler = StandardScaler()
scaler.fit(p_df_tr[['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']])
p_x = scaler.transform(p_df[['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']])
p_y = np.array([1 if t == 'Yes' else 0 for t in p_df['type']])

p_x_tr = p_x[0:len(p_df_tr)]
p_y_tr = p_y[0:len(p_df_tr):]
p_x_te = p_x[len(p_df_tr):]
p_y_te = p_y[len(p_df_tr):]

## %% Prepare KNN classifier

def knn_holdout(x_tr, y_tr, x_te, y_te, max_k):
    score = np.zeros(max_k)
    for i in range(max_k):
        k = i + 1
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_tr, y_tr)
        score[i] = knn.score(x_te, y_te)
    return score

def knn_loo(x, y, max_k):
    score = np.zeros(max_k)
    for i in range(max_k):
        k = i + 1
        knn = KNeighborsClassifier(n_neighbors=k)
        loo = LeaveOneOut()
        scores = cross_val_score(knn, x, y=y, cv=loo, n_jobs=-1, verbose=1)
        score[i] = scores.mean()
    return score

# %% Classification

pima_holdout_scores = knn_holdout(p_x_tr, p_y_tr, p_x_te, p_y_te, P_K_MAX)
pima_loo_scores = knn_loo(p_x, p_y, P_K_MAX)
etl1_holdout_scores = knn_holdout(e_x_tr, e_y_tr, e_x_te, e_y_te, E_K_MAX)
etl1_loo_scores = knn_loo(e_x, e_y, E_K_MAX)

# %% Plot Pima indian

plt.scatter(range(1, E_K_MAX + 1), 1 - etl1_holdout_scores, marker='x', label='Hold-Out')
plt.scatter(range(1, E_K_MAX + 1), 1 - etl1_loo_scores, marker='o', label='Leave-One-Out')
plt.xlabel('k')
plt.ylabel('error')
plt.title('ETL1 dataset')
plt.legend()
plt.show()

# %% Plot Pima indian

plt.scatter(range(1, P_K_MAX + 1), 1 - pima_holdout_scores, marker='x', label='Hold-Out')
plt.scatter(range(1, P_K_MAX + 1), 1 - pima_loo_scores, marker='o', label='Leave-One-Out')
plt.xlabel('k')
plt.ylabel('error')
plt.title('Pima indian dataset')
plt.legend()
plt.show()
