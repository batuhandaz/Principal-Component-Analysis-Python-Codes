#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:18:16 2021

@author: batuhan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
X,y = datasets.load_wine(return_X_y=True)
X.shape

X_df = pd.DataFrame(X)
X_df.corr(method = 'pearson')

correlation, pvalue = pearsonr(X[:,5], X[:,6])
print(correlation)

plt.scatter(X[:,5], X[:,6])
plt.show

model = PCA()
transformed = model.fit_transform(X)

print(transformed.shape)

transformed_df = pd.DataFrame(transformed)
transformed_df.corr(method = 'pearson')

plt.scatter(transformed[:,5], transformed[:,6])
plt.show

scaler = StandardScaler()
pipeline = make_pipeline(scaler, model)
pipeline.fit(X)

pca_features= PCA(n_components = 3)
dimension_reduction_result = pca_features.fit_transform(X)
dimension_reduction_result.shape 