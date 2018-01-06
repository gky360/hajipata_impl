# -*- coding: utf-8 -*-

import struct

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

# %% Load image

H, W = 63, 64


def load_etl1_image(filename, idx):
    with open(filename, 'rb') as f:
        f.seek(idx * 2052)
        s = f.read(2052)
        r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
        iP = iF.convert('P')
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        org_image = np.array(iE.getdata(), dtype=np.uint8).reshape(H, W)
        digit = int(r[3]) % 0x10
    return org_image, digit


filename = './data/ETL1/ETL1C_01'
x, y = load_etl1_image(filename, 7000)
x = x.ravel() // 16
hist = np.bincount(x, minlength=16)

# %% Plot histogram


# %% Plot interclass variance

N = H * W


def calc_interclass_var(thresh):
    mask = x > thresh
    N2 = np.count_nonzero(mask)
    N1 = N - N2
    if N1 == 0 or N2 == 0:
        return 0.0
    m = x.mean()
    m1 = x[np.invert(mask)].mean()
    m2 = x[mask].mean()
    sig = (N1 * ((m1 - m) ** 2) + N2 * ((m2 - m) ** 2)) / N
    return sig


v = np.array([calc_interclass_var(th) for th in range(16)])

fig, ax = plt.subplots(2, 2, figsize=(8, 6))

ax[0, 0].imshow(x.reshape(H, W), cmap='gray')
ax[0, 0].axis('off')

ax[0, 1].plot(hist)
ax[0, 1].set_xlabel('Value')
ax[0, 1].set_xlim(0, 15)
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].set_ylim(0, None)

ax[1, 0].plot(v)
ax[1, 0].set_xlabel('Value')
ax[1, 0].set_xlim(0, 15)
ax[1, 0].set_ylabel('Inter-class variance')
ax[1, 0].set_ylim(0, None)

ax[1, 1].imshow((x > v.argmax()).reshape(H, W), cmap='gray')
ax[1, 1].axis('off')

plt.tight_layout()
plt.savefig('./chap06/6_2_discriminant_analysis_otsu.png')
plt.show()

# %% Plot Otsu

img = x.reshape(H, W) * 16
thr, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fix, ax = plt.subplots(figsize=(3, 3))
ax.imshow(mask, cmap='gray')
ax.axis('off')
plt.show()
