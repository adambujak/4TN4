import numpy as np
import pandas as pd

from sklearn.utils import shuffle
def shuffle_data(X, y):
    X, y = shuffle(X, y, random_state=0)
    return X, y

#2 Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,0:18].values.astype(float)
y = dataset.iloc[:,18].values.astype(float)

print(X)
print(y)
X, y = shuffle_data(X,y)

X = X[0:10000]
y = y[0:10000]
print(X)
print(y)

#3 Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)

#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussiantype) kernel.

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=int(0.2*len(X)), random_state=4)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

from joblib import dump
#dump(sc_X, 'std_scaler_x.bin', compress=True)
#dump(sc_y, 'std_scaler_y.bin', compress=True)
dump(regressor, 'regressor.joblib')

print("model score: ", regressor.score(x_test, y_test))

