# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:18:44 2017

@author: 317005
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

# Importing the dataset
dataset = pd.read_table('data.txt')
X = dataset.values[:,:3]
y = dataset.values[:,3]

neg = np.count_nonzero(y == -1)
pos = np.count_nonzero(y == 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    c = 'r' if y[i] == 1 else 'b'
    ax.scatter(X[i][0], X[i][1], X[i][2], c=c, marker='o')

ax.set_xlabel('Colesterol total')
ax.set_ylabel('Idade')
ax.set_zlabel('Glicemia sérica em jejum')

plt.show()

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(321)
plt.title("Colesterol total", fontsize='small')
plt.boxplot(X[:, 0])

plt.subplot(322)
plt.title("Colesterol total x Idade", fontsize='small')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

plt.subplot(323)
plt.title("Idade", fontsize='small')
plt.boxplot(X[:, 1])

plt.subplot(324)
plt.title("Colesterol total x Glicemia sérica em jejum", fontsize='small')
plt.scatter(X[:, 0], X[:, 2], marker='o', c=y)

plt.subplot(325)
plt.title("Glicemia sérica em jejum", fontsize='small')
plt.boxplot(X[:, 2])

plt.subplot(326)
plt.title("Idade x Glicemia sérica em jejum", fontsize='small')
plt.scatter(X[:, 1], X[:, 2], marker='o', c=y)

#- 1 purple
# 1 yellow
