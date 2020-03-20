# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting linear regression model

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)


#fitting polynomial regressor to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X) 

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


#visualizing the linear regressoin results

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or Bluff(for linear regression)")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()


#visualizing the polynomial regression results


plt.scatter(X,y,color='red')
#plt.plot(X,lin_reg2.predict(X_poly),color='blue') or we can make it suitable for all other type
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X) ),color='blue')
plt.title("Truth or Bluff(for polynomial regression)")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()


#predict the new result with linear regression


lin_reg.predict(6.5)


#predict the new result with polynomial regression

lin_reg2.predict(poly_reg.fit_transform(6.5))



