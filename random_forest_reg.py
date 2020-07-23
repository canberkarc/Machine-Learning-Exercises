# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:49:34 2020

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

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#visualization
plt.scatter(x,y, color = "red")
plt.plot(x_, y_head, color = "green")
plt.xlabel("Tribune level")
plt.ylabel("Price")
plt.show()