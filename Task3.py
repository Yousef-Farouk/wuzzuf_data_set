# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 20:19:54 2022

@author: yfrou
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 0: 8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define  k-NN
classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')

# fit model
classifier.fit(X_train,y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
logistic_model = LogisticRegression(fit_intercept=True,C=1e15)
logistic_model.fit(X_train,y_train)
predicted = logistic_model.predict(X_test)

