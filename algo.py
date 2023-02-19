### Task 1.1

import numpy as np
import random as random
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


### Task 1.1
data = pd.read_csv('kc_house_data.csv', delimiter=",")

precio = data['price']
living = data['sqft_living']

precio = np.array(data['price'])
living = np.array(data['sqft_living'])
x_train, x_test, y_train, y_test = train_test_split(living, 
                                                    precio, 
                                                    test_size=0.2,
                                                    random_state=1234)
##-------------------------------------------
def create_matrices(polynomial_degree):
    temp_x =[]
    temp_w =[]
    temp_y = []
    x = np.array([])
    w = np.array([])
    y = np.array([])
    for i in range(0, polynomial_degree + 1):
        temp_x.append(x_train**i)
        temp_w.append([random.randint(1, 10)])
    
    for element in y_train:
        temp_y.append([element])

    return np.array(temp_x), np.array(temp_w), np.array(temp_y)
### Task 1.2

polynomial_degree = 3

x, w, Y = create_matrices(polynomial_degree)  

X = x.transpose()

##------------Fin Task 1.2------------

iterator = 5000
i = 0
error = 0
# L = 0.00000000000001
L="0.0"

L += (3**(polynomial_degree + 2))*"0"+"1"
L = float(L)

# Valor para polinomio 2
# L = 0.000000000000001

# Valor para polinomio 3
# L = 0.0000000000000000000001


n = len(x_train)


while i < iterator:
    
    # t = w.transpose() @ x
    
    error = (1.0/n) * ((Y - X @ w).transpose() @ (Y - X @ w))

    w = w - L * ((-2/n) * (x @ (Y - X @ w)))
    #w = w - (L/n) * (X.T @ (X @ w - t))


    i += 1

print("Error : ",error[0][0])

for i in w:
    print(i)

# x -> x_train
print(w)
print(len(X))


y_pred = (X @ w).transpose()
print(y_pred[0])


mymodel = np.poly1d(np.polyfit(x_train, y_pred[0], polynomial_degree))

myline = np.linspace(1, 14000)

sortedx = np.sort(x_train)
sortedy = np.sort(y_pred[0])

plt.scatter(x_train, y_train)
plt.plot(myline, mymodel(myline))
plt.show()
"""
plt.plot(sortedx, sortedy, color = 'green')
plt.scatter(x_train, y_train) 
# plt.plot([min(x_train), max(x_train)], [min(ylist), max(ylist)], color='red')  # regression line
plt.show()"""