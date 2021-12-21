# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:03:35 2021

@author: yfrou
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
dataset=pd.read_csv("Wuzzuf.csv")
X = pd.DataFrame(dataset.iloc[:, :].values)

#Display structure and summary of the data.
dataset.describe()
dataset.info()
# dropping ALL duplicate values
dataset.drop_duplicates(keep = "first", inplace = True)

# sorting by Title
dataset.sort_values("Title", inplace = True)

# Taking care of missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer = imputer.fit(X.iloc[:,:])
X.iloc[:,:] = imputer.transform(X.iloc[:,:])


# What are the most demanding companies for jobs?
companies=dataset["Company"].value_counts().head(10)
mylabels=list(companies.keys())
plt.pie(companies,labels=mylabels)

# What are it the most popular job titles?
jobs=dataset["Title"].value_counts().head(10)

#bar chart
values=list(jobs.keys())
plt.subplots(figsize =(16, 9))
plt.bar(values ,jobs, color ='maroon',width = 0.3)
plt.xticks(size="x-small")
plt.show()

# What are the most popular areas?
areas=dataset["Country"].value_counts().head(10)
v=list(areas.keys())
#plt.subplots(figsize =(16, 9))
fig = plt.figure(figsize = (16, 9))
plt.bar(v ,areas, color ='maroon',width = 0.4)
plt.xticks(size="x-small")
plt.show()

#skills
s=dataset["Skills"].value_counts().head(11)
print(s)

















