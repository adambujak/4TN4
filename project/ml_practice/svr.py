#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values.astype(float)
y = dataset.iloc[:,2:3].values.astype(float)

new_x = []
for x in X:
    new_x += [[x[0],x[0]]]
X = new_x
print(X)


#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here
from sklearn.svm import SVR
# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussiantype) kernel.
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

from joblib import dump
dump(sc_X, 'std_scaler_x.bin', compress=True)
dump(sc_y, 'std_scaler_y.bin', compress=True)
dump(regressor, 'regressor.joblib')


#5 Predicting a new result
#in_trans = sc_X.transform(np.array([[6.5, 6.5], [3.5, 3.5]]))
#y_pred = regressor.predict(in_trans);
#out_pred = []
#for pred in y_pred:
#    out_pred += [[pred]]
#y_pred = out_pred
#y_pred = sc_y.inverse_transform(y_pred)
#print(y_pred)
