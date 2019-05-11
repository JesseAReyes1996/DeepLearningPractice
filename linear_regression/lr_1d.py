import numpy as np
import matplotlib.pyplot as plt

#read the data
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

#convert arrays into numpy arrays
X = np.array(X)
Y = np.array(Y)

#calculate a and b, derived from formula
denominator = np.dot(X, X) - X.mean() * X.sum()

a = (np.dot(X, Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * np.dot(X, X) - X.mean() * np.dot(X, Y)) / denominator

#calculate predicted-y
yHat = a*X + b

plt.scatter(X,Y)
plt.plot(X, yHat)
plt.show()

#calculate R^2
residual = Y - yHat
total = Y - Y.mean()

#R^2 = 1 - (SSresidual / SStotal)
R2 = 1 - (np.dot(residual, residual) / np.dot(total,total))
