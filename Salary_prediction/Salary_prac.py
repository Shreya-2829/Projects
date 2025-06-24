# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 18:08:14 2025

@author: shreya
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv(r"D:\nit_prac\assignments\salary_pred\Salary_Data.csv")


#  .iloc - Integer-location based indexing.
# Allows you to select rows and columns by position (not by name).

x=dataset.iloc[:,:-1]

y=dataset.iloc[:,-1]


# training and testing phase always works on historical data (Seen data)
# where as validation phase always works on future data (seen data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2,train_size=0.8,random_state=0)


from sklearn.linear_model import LinearRegression

# 👆 It allows you to create a linear regression model, which tries to find the best-fitting straight line through your data.


# 👇  regressor - Ordinary least squares Linear Regression.  LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
regressor=LinearRegression()

# You’re creating an instance (object) of the LinearRegression model.  --- At this point, the model is created, but not trained yet.

regressor.fit(x_train, y_train)

# .fit() means train the model. --> This is where the learning happens — the model calculates the slope and intercept.

y_pred=regressor.predict(x_test)

# .predict() uses the trained model to make predictions. 

# x_test: the test data (new inputs).   --------  y_pred: the predicted output values for x_test.

# 📌 This is where the model applies what it learned to new data.

# compare predicted and actual salariesfrom test set
comparision=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparision)

# pd.DataFrame({...})  ----  Creates a new Pandas DataFrame. A DataFrame is like a table — rows and columns — where you can easily view and analyze data.
# {'Actual': y_test, 'Predicted': y_pred}  -----  A dictionary that defines two columns:

# 'Actual': Contains the true target values (y_test)
# 'Predicted': Contains the model's predicted values (y_pred)


plt.scatter(x_test, y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs exp')
plt.ylabel('exp')
plt.xlabel('salary')
plt.show()


m_slope=regressor.coef_   # .coef_  : An attribute of the model that stores the slope(s) (also called coefficients).
print(m_slope)

c_intercept=regressor.intercept_   #  .intercept_  : The y-intercept of the regression line — the value of y when x = 0.
print(c_intercept)

#predicting future using m and c 👇

y_12=m_slope*12+c_intercept
print(y_12)


#  When you use .score(X, y) on a regression model in scikit-learn (like LinearRegression), it returns the R² score (also called coefficient of determination).
#  R² Score (Coefficient of Determination)
#  ⭐⭐⭐ The R² score tells you how well your regression model fits the actual data.

bias=regressor.score(x_train,y_train)  # How well your model fits the training data.
print("bias",bias)
 
variance=regressor.score(x_test,y_test)  # How well the model generalizes to unseen data.
print("variance",variance)

# stats 


dsm=dataset.mean()  # gives mean of each attribute of dataset
print(dsm)

sm=dataset['Salary'].mean() 
print(sm)

dsmed=dataset.median()
print(dsmed)

smed=dataset['Salary'].median()
print(smed)

print("mode")
smod=dataset['YearsExperience'].mode()
print(smod)

var=dataset.var()
print(var)

svar=dataset['Salary'].var()
print(svar)

dsstd=dataset.std()
print(dsstd)

salstd=dataset['Salary'].std()
print(salstd)

#co-efficient of variations = standard deviation / mean

from scipy.stats import variation
dacv=variation(dataset.values)  # this will give u cv of entire dataframe
print(dacv)
salcv=variation(dataset['Salary'].values)
print(salcv)


# Correlation

#  --->  Correlation Matrix?
# A correlation matrix is a table that shows the relationship (correlation) between all pairs of variables (columns) in a dataset.
# Each cell in the table tells you how much one variable is related to another.

dscorr=dataset.corr()  # gives correlation of entire dataset
print(dscorr)
# in correlations diagonals are always 1
 
 
bwcorr=dataset['Salary'].corr(dataset['YearsExperience'])  # this gives correlation between these 2 attributes 
print(bwcorr)


# Skewness - tells you whether the data is symmetrical or not.
#Positive skew = tail on right, Negative skew = tail on left.

dsskew=dataset.skew()
print(dsskew)


salskew=dataset['Salary'].skew()
print(salskew)


# standard error of Mean (SEM) = standard deviation / √n 

dsstd=dataset.sem()  # this will give standard error across all columns
print(dsstd)

salstd=dataset['Salary'].sem()
print(salstd)

# Z-score -  standardizes data (mean=0, std=1).
# Measures how many standard deviations a value is from the mean.

import scipy.stats as stats
dszscore=dataset.apply(stats.zscore) # zscore of entire dataframe
print(dszscore)


salzscore=stats.zscore(dataset['Salary'])
print(salzscore)

# Degree of freedom
a = dataset.shape[0]  # this will give us no.of rows 
b = dataset.shape[1]  # this will give us no.of cols 

dof=a-b  # dof of entire dataset
print(dof)


# sum of square regressor (SSR) - Measures explained variance: how much the model explains the variation in the data.

y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)

print(SSR)

# Sum of Squared Errors (SSE) - Measures unexplained variance: difference between actual and predicted
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

# Total Sum of Squares (SST) - Measures total variance in data.
mean_Total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_Total)**2)
print(SST)


# R2 square --> [R² Score (Coefficient of Determination)] = how much variance the model explains.
#   R² = 1 → perfect model;   R² = 0 → model explains nothing.


r_square = 1- (SSR/SST)
print(r_square)


# save trained model to disk

import pickle
filename='linear_regression_model.pk1'
with open(filename, 'wb') as file:   # 'wb' means write in binary mode
    pickle.dump(regressor,file)
print("model has been pickled and saved as linear_regression_model.pk1")


#Prints the current working directory (where the model is saved).

import os
print(os.getcwd())












