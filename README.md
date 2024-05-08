# EX 02: Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## DATE:

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 

2. Set variables for assigning dataset values. 
3. Import linear regression from sklearn. 4. Assign the points for representing in the graph. 
5. Predict the regression for marks by using the representation of the graph. 
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VENKATANATHAN P R
RegisterNumber: 212223240173
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import libraries to find mae, mse

from sklearn.metrics import mean_absolute_error,mean_squared_error

# Read csv file

df=pd.read_csv("student_scores.csv")

# Displaying the content in datafile

print("HEAD:")
print(df.head())
print("TAIL:")
print(df.tail())

# Segregating data to variables

x=df.iloc[:,:-1].values
print("x values")
print(x)

y=df.iloc[:,1].values
print("y values")
print(y)

# Splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# Displaying predicted values

print("Predicted values")
print(Y_pred)

# Displaying actual values

print("Actual values")
print(Y_test)

# Graph plot for training data

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Graph plot for test data

plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Find MAE,MSE,RMSE

MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
```

## Output:
![alt text](<Screenshot 2024-03-22 213038.png>)
![alt text](<Screenshot 2024-03-22 214525.png>)
![alt text](<Screenshot 2024-03-22 214531.png>) 
![alt text](image-1.png)
![alt text](image.png)
![alt text](<Screenshot 2024-03-22 213140.png>) ![alt text](<Screenshot 2024-03-22 213149.png>)
![alt text](<Screenshot 2024-03-22 213157.png>)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
