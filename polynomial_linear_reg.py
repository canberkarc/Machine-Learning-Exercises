# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:11:01 2020

@author: canberk
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\canbe\Downloads\polynomial_regression.csv",sep = ";")

x = df.price.values.reshape(-1,1)
y = df.max_speed.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Max Speed")
plt.xlabel("Price")
plt.show()

# Linear Regression =  y = b0 + b1*x
# Multiple Linear Regression   y = b0 + b1*x1 + b2*x2

# linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

# Prediction
y_head = lr.predict(x)

plt.plot(x,y_head,color="red",label ="linear")
plt.show()

print("Price prediction for a car whose value is 10000$: ",lr.predict([[10000]]))

# As it can be seen that linear regression is not a proper choice to make prediction on such
# curvilinear data becuase linear regression find a car's max speed as 871 km/h

#Polynomial Linear Regression is a better choice

#Getting x^2 expression 
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)
# x value whihc is price becomes x^2
x_polynomial = polynomial_regression.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_polynomial, y)

#Prediction with new x value and initial y value
y_head2 = lr2.predict(x_polynomial)

plt.plot(x,y_head2, color = "green", label = "poly")
plt.legend()
plt.show()