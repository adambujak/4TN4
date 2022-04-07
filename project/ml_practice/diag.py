import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_image(lines):
    out_array = []
    for line in lines:
        numbers = line.split(",")
        for number in numbers:
            try:
                num = int(number)
                out_array += [num]
            except ValueError:
                pass

    return out_array

def import_data(file):
    with open('diag_train/{}'.format(file)) as f:
        lines = f.readlines()
    in_data = lines[0:3]
    out_data = lines[4]

    in_image = make_image(in_data)
    out_image = int(out_data.split('\n')[0])

    print(in_image)
    print(out_image)
    return in_image, out_image

def append_dataset(file, X, y):
    tX, ty = import_data(file)
    X += [tX]
    y += [ty]

X = []
y = []

append_dataset("1", X, y)
append_dataset("2", X, y)
append_dataset("3", X, y)
append_dataset("4", X, y)
append_dataset("5", X, y)
append_dataset("6", X, y)

X = np.array(X)
y = np.array(y)

##3 Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)

#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here
from sklearn.svm import SVR
# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussiantype) kernel.
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#5 Predicting a new result
#y_pred = regressor.predict([[6.5]])
in_image, out_image = import_data("8")
in_image = np.array([in_image])

print(in_image)

y_pred = regressor.predict(in_image);
print(y_pred)
