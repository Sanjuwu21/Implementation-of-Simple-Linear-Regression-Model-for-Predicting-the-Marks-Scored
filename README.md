# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1 Import necessary libraries (e.g., pandas, numpy,matplotlib).
2 Load the dataset and then split the dataset into training and testing sets using sklearn library.
3 Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4 Use the trained model to predict marks based on study hours in the test dataset.
5 Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sanjeev D
RegisterNumber: 212223040185
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```
## Output:
## 1)HEAD:
![image](https://github.com/Sanjuwu21/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146498969/7d408326-1825-4435-9579-35b89f942d56)

## 2)GRAPH OF PLOTTED DATA:
![image](https://github.com/Sanjuwu21/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146498969/da89f7e0-d3c0-4878-9794-fbb1c944cea4)

## 3)TRAINED DATA:
![image](https://github.com/Sanjuwu21/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146498969/742f41e1-4835-4236-9dd1-d4762c324ea8)

## 4)LINE OF REGRESSION:
![image](https://github.com/Sanjuwu21/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146498969/992c6e6e-ffc2-4fce-980c-897aeaaa66c7)

## 5)COEFFICIENT AND INTERCEPT VALUES:
![image](https://github.com/Sanjuwu21/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146498969/6fa56ee7-139a-4457-a4b5-4e20813bd39f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
