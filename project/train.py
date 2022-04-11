import numpy as np
import pandas as pd

from sklearn.utils import shuffle
def shuffle_data(X, y):
    X, y = shuffle(X, y, random_state=0)
    return X, y

# import data
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,0:18].values.astype(float)
y = dataset.iloc[:,18].values.astype(float)

X, y = shuffle_data(X,y)

X = X[0:10000]
y = y[0:10000]

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=int(0.2*len(X)), random_state=4)

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)

from joblib import dump
# save model
dump(regressor, 'regressor.joblib')

print("model score: ", regressor.score(x_test, y_test))

