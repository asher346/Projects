#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y ,test_size=0.2,random_state=0)

#Training
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

#Predicting test set result
Y_pred = lr.predict(X_test)

#Visualising training set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.title('Salary Vs Exerience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()