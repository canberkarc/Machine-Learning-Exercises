# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:52:43 2020

@author: canberk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\canbe\Downloads\data.csv")
print(data.info())
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

#Applying transpose to data to make pixel numbers start from lowest
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#Parameter initalize and Sigmoid Function
#For our data,dimension is 30 because there 30 features
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

w,b = initialize_weights_and_bias(30)

#Sigmoid Function
def sigmoid(z):
    y_head = 1 / (1 + (np.exp(-z)))
    return y_head


#Forward and Backward Propagation
def forward_backward_propagation(w,b,x_train,y_train):
    #Forward Propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head) 
    cost = (np.sum(loss)) / x_train.shape[1] #x_train.shape[1] is for scaling
    
    #Backward Propagation
    derivative_weight = (np.dot(x_train,((y_head - y_train).T))) / x_train.shape[1] #x_train.shape[1] is for scaling
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] #x_train.shape[1] is for scaling
    gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
    
    return cost,gradients

#Updating (learning) parameters
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iteration):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
    
        if i%10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i : %f" %(i,cost))
        
    #I update(learn) parameters weights and bias
    parameters = {"weight" : w, "bias" : b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters,gradients,cost_list

#Prediction Function
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    #Threshold is 0.5
    #If x is greater than threshold then prediction is sign one(y_head = 1)
    #If x is smaller than threshold then prediction is sign zero(y_head = 0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
            
    return y_prediction

#Logistic Regression
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iteration):
    dimension = x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    
    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,num_iteration)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Accuracy of test: {}".format(100 - (np.mean(np.abs(y_prediction_test - y_test)) * 100)))
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iteration = 500)
#It can be seen that the more iteration the more accuracy
