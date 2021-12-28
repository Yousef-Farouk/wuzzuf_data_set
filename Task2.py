# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:46:40 2021

@author: yfrou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Wuzzuf.csv')
X = pd.DataFrame(dataset.iloc[:, [0, 1]].values)
x=pd.DataFrame(X)

# fatorize 
dataset['fact'] = pd.factorize(dataset['YearsExp'])[0]

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
labelEncoder.fit(X.iloc[:,0])
labelEncoder.fit(X.iloc[:,1])
X.iloc[:,0]= labelEncoder.transform(X.iloc[:,0])
X.iloc[:,1]= labelEncoder.transform(X.iloc[:,1])
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X.iloc[:,[0,1]])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#how many clusters 
from kneed import KneeLocator
k1 = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
print(k1.elbow)
        
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
X = np.array(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'yellow', label = 'Centroids')
plt.show()