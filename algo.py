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

### Task 1.2
x = np.array([x_train**0,
              x_train,
              x_train**2,
              x_train**3,
              x_train**4])       

X = x.transpose()

w = np.array([[random.randint(1,10)], 
                [random.randint(1,10)],
                [random.randint(1,10)],
                [random.randint(1,10)]])

##------------Fin Task 1.2------------

#aa=np.array([1,2],[2,1])
#bb=np.array([3,3])

print(X[0] * w)

# Task 1.3
'''
Y_predicted = (w_1 * X + w_0)

iterations = 1000
last_error = 1000
i = 0

error = 1

L = 0.0000001
while (i < iterations and error > 0.005):
    Y_predicted = w_1 * X + w_0
    
    Dm = (-2.0/n)*sum(X*(Y - Y_predicted))
    Dc = (-2.0/n)*sum(Y - Y_predicted)
    
    suma = 0
    for i in range(len(Y)):
        suma += (Y[i] - Y_predicted[i]) ** 2
    error = (1.0/n) * suma

    w_0 = w_0 - L * Dc
    w_1 = w_1 - L * Dm

    if last_error > error:
        last_error = error

    i+=1

Y_pred = w_1 * X + w_0

print("M = ",w_1)
print("C = ",w_0)

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()
'''