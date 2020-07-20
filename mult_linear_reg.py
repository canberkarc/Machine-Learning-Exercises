# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:14:15 2020

@author: canberk
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\TEMP\Desktop\multiple_linear_regression_dataset.csv", sep = ";")

x = df.iloc[:,[0,2]].values
y = df.salary.values.reshape(-1,1)

#model
mult_linear_reg = LinearRegression()

#fitting values to the model
mult_linear_reg.fit(x,y)

print("b0 : ",mult_linear_reg.intercept_)
print("b1 : , b2 : ", mult_linear_reg.coef_)

print("Salary prediction according to experience = 10 years,age = 35 and experience = 5 years, age = 35 ", mult_linear_reg.predict(np.array([[10,35],[5,35]])))
