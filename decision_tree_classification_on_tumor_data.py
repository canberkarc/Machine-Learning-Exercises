# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 01:09:58 2020

@author: canberk
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"C:\Users\canbe\Desktop\data.csv")


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  
# benign = B


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Malignant",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Benign",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

#Naive Bayes Model
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)

#Show score
print("Score: ",dtc.score(x_test,y_test))
#Score:  0.9181286549707602