import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"C:\Users\TEMP\Desktop\a.csv",sep = ";")

#plt.scatter(df.deneyim,df.maas)
#plt.xlabel("deneyim")
#plt.ylabel("maas")
#plt.show()

#import linear regression
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

b0 = linear_reg.predict([[0]])
print("b0 : ", b0)

b0 = linear_reg.intercept_
print("b0 : ", b0)

b1 = linear_reg.coef_
print("b1 : ", b1)

# maas = 1663 + 1138*deneyim

yeni_maas = 1663 + 1138*11
print("Calculation : ", yeni_maas)

predict_yeni_maas = linear_reg.predict([[11]])
print("Prediction : ", predict_yeni_maas)

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,21]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)

plt.plot(y_head, color = "purple")