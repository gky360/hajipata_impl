# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# In[]:
iris = datasets.load_iris()
data = iris.data
print(iris.feature_names)
print(data)


# In[]:
data.mean(axis=0)
np.cov(data, rowvar=0)

# In[]:
data_std = (data - data.mean(axis=0)) / data.std(axis=0)
data_std.mean(axis=0)

# In[]:
plt.figure(figsize=(6, 6))
plt.scatter(data[0:50, 2], data[0:50, 3], marker='$s$')
plt.scatter(data[50:100, 2], data[50:100, 3], marker='$c$')
plt.scatter(data[100:150, 2], data[100:150, 3], marker='$v$')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Before standardization')
plt.show()

# In[]:
plt.figure(figsize=(6, 6))
plt.scatter(data_std[0:50, 2], data_std[0:50, 3], marker='$s$')
plt.scatter(data_std[50:100, 2], data_std[50:100, 3], marker='$c$')
plt.scatter(data_std[100:150, 2], data_std[100:150, 3], marker='$v$')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('After standardization')
plt.show()
