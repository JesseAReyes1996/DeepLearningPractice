import re
import numpy as np
import matplotlib.pyplot as plt

#read the data
X = []
Y = []

nonDecimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')
    #get the year
    x = int(nonDecimal.sub('', r[2].split('[')[0]))
    #get the number of transistors per year
    y = int(nonDecimal.sub('', r[1].split('[')[0]))

    X.append(x)
    Y.append(y)

#convert arrays into numpy arrays
X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

Y = np.log(Y)
plt.scatter(X,Y)
plt.show()

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
print("a: " , a)
print("b: " , b)
print("R2: " , R2)

#transistorCount2 = 2 * transistorCount1
#ln(transistorCount2) = ln(2 * transistorCount1)
#ln(transistorCount2) = ln(2) + ln(transistorCount1)

#substitue transistorCount = a * year + b to transistorCount1 and transistorCount2

#a * year2 + b = ln(2) + a * year1 + b
#a * year2 = ln(2) + a * year1
#year2 = ln(2) / a + year1

#year2 is the time (in years) that it will have taken for the number of
#transistors to have doubled

year2 = np.log(2) / a
print('time taken to double transistor count(in years): ' , year2)
