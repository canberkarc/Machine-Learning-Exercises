# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 01:55:27 2020

@author: canberk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\canbe\Downloads\random_forest_regression_dataset.csv", sep = ";", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)

y_head = rf.predict(x)

#Evaluating Performance with R-Square Method

from sklearn.metrics import r2_score

print("R-Square Score : ",r2_score(y,y_head))