from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

sc_X = load('std_scaler_x.bin')
sc_y = load('std_scaler_y.bin')
regressor = load('regressor.joblib')

in_trans = sc_X.transform(np.array([[6.5, 6.5], [3.5, 3.5]]))
y_pred = regressor.predict(in_trans);
out_pred = []
for pred in y_pred:
    out_pred += [[pred]]
y_pred = out_pred
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

