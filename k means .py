# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:39:15 2019

@author: dhruv
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

Iris = datasets.load_iris()
df = pd.DataFrame(Iris.data, columns = Iris.feature_names)
Tar = pd.DataFrame(Iris.target, columns=['Tar'])
df = pd.concat([df, Tar], join='inner', axis = 1)

v = df.iloc[:, [0, 1, 2, 3, 4]].values

classes = []

rgd = range(1, 10)
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(v)
    classes.append(kmeans.inertia_)

plt.plot(rgd, classes, 'o-')
plt.xlabel('K')
plt.ylabel('Cluster')
plt.title('Elbow heuristic at 3')
plt.show()