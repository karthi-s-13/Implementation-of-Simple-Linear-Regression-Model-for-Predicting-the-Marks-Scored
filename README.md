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
### DATASTE
![Screenshot 2025-05-02 200538](https://github.com/user-attachments/assets/9942f654-9267-44d3-bc91-2e5feb73218f)

### X Value
![Screenshot 2025-05-02 200549](https://github.com/user-attachments/assets/20eb3d65-c9f6-42c0-8ea4-58d2d348efc6)

### Y Value
![Screenshot 2025-05-02 200559](https://github.com/user-attachments/assets/63bd5bcf-5096-4f6e-bc72-953cc0b62eeb)

### Predict Value
![Screenshot 2025-05-02 200611](https://github.com/user-attachments/assets/bfb7fa00-4aba-4c1e-87e7-81eff6258ca1)

### Target Test Value
![Screenshot 2025-05-02 200617](https://github.com/user-attachments/assets/aeee8897-bc26-4d59-b860-ad98d1e81bd0)

### Training Set
![Screenshot 2025-05-02 200636](https://github.com/user-attachments/assets/c73c728c-4e82-4210-b146-835f0e2eebbc)

### Testing Set
![Screenshot 2025-05-02 200648](https://github.com/user-attachments/assets/796429bd-29eb-4212-8fcd-0585cad63ad0)

### Mean Square Error
![Screenshot 2025-05-02 200659](https://github.com/user-attachments/assets/15a76f9a-c7b7-41bc-9836-699dea17abb6)

### Mean Absolute Error
![Screenshot 2025-05-02 200706](https://github.com/user-attachments/assets/6ed82f1c-2c5c-4960-b3f5-8a20198c4d68)

### Root Mean Square Error
![Screenshot 2025-05-02 200712](https://github.com/user-attachments/assets/e91974d3-1cb8-4988-b22f-38fe688e5e16)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
