import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  

dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')

x = dataset["GRE Score"]
x = np.array(x).reshape((-1, 1)) 
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head())