# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHIKEYAN S
RegisterNumber:  212224230116
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
```
```
data = pd.read_csv("/content/drive/MyDrive/Data set/student_scores.csv")
df= pd.DataFrame(data)
df
```
```
x = df.iloc[:,:-1].values
x
```
```
y = df.iloc[:,-1].values
y
```
```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
```
```
reg = LinearRegression()
reg.fit(x_train,y_train)
```
```
predict =reg.predict(x_test)
```
```
predict
```
```
y_test
```
```
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Hours vs Percentage')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()
```
```
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,predict,color='blue')
plt.title('Hours vs Percentage')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()
```
```
mse = mean_squared_error(y_test,predict)
print("Mean Square Error: {:.2f}".format(mse))
```
```
msa = mean_absolute_error(y_test,predict)
print("Mean Absolute Error: {:.2f}".format(msa))
```
```
sq = np.sqrt(mse)
print("Root Mean Square Error: {:.2f}".format(sq))
```


## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
