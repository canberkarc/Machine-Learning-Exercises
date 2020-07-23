# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:24:16 2020

@author: canberk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\canbe\Downloads\decision_tree_regression_dataset.csv", sep = ";",header = None)

x = df.iloc[:, 0].values.reshape(-1,1)
y = df.iloc[:, 1].values.reshape(-1,1)

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

#Prediction
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

#visualization
plt.scatter(x,y, color = "turquoise")
plt.plot(x_, y_head, color = "green")
plt.xlabel("Tribune Level")
plt.ylabel("Price")
plt.show()