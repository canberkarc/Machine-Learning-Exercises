# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 01:31:12 2020

@author: canberk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r"C:\Users\canbe\Downloads\data.csv")
#dropping unnecessary columns
data.drop(["Unnamed: 32","id"],axis = 1, inplace = True)

#in column diagnosis M means malignant tumor, B means benign tumor
#to use this feature I change it to numerical feature

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#Normalization
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

#Split 20% of data for test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Test Accuracy : {}".format(lr.score(x_test,y_test)))


