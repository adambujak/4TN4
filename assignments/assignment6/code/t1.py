import numpy as np

data = [[0,0,0,0,0,0,0,0,0,0],
       [0,255,255,255,0,0,0,0,0,0],
       [0,255,0,255,0,0,0,0,0,0],
       [0,255,255,255,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,255,0,0,255,0],
       [0,0,0,0,0,255,0,0,255,0],
       [0,0,0,0,0,255,0,0,255,0],
       [0,0,0,0,0,255,255,255,255,0],
       [0,0,0,0,0,0,0,0,0,0]]


d = 10
theta_vals = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

acc = np.zeros((d, len(theta_vals)))

for i in range(len(data)):
    for j in range(len(data[i])):
        current_pixel = data[i][j]
        if current_pixel == 255:
            for k in range(len(theta_vals)):
                theta = theta_vals[k]
                theta = theta *3.14/180
                d = j*np.cos(theta) + i*np.sin(theta)
                if(d<0):
                    continue
                try:
                    acc[int(d)][k] += 1
                except:
                    print("error", theta, d);

print(acc)


