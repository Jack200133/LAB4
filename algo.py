### Task 1.1

import numpy as np
import random as random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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

random.seed(1234)
##-------------------------------------------

### Task 1.2
def create_matrices(polynomial_degree, x_train, y_train):
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
##------------Fin Task 1.2------------

##------------Task 1.3------------
def gradient_descent(polynomial_degree, x_train, y_train):
    
    x, w, Y = create_matrices(polynomial_degree, x_train, y_train)  

    X = x.transpose()
    iterator = 5000
    i = 0
    error = 0

    L =[0,0.00000001,0.0000000000000000000001,0.00000000000000000000000000001]

    L = L[polynomial_degree]

    n = len(x_train)


    while i < iterator:
        
        
        error = (1.0/n) * ((Y - X @ w).transpose() @ (Y - X @ w))

        w = w - L * ((-2/n) * (x @ (Y - X @ w)))
        i += 1

    return X, w, error[0][0]

polynomial_degree = 1

X, w, error = gradient_descent(polynomial_degree,x_train, y_train)

y_pred = (X @ w).transpose()

sortedx = np.sort(x_train)
sortedy = np.sort(y_pred[0])

plt.plot(sortedx, sortedy, color = 'red')
plt.scatter(x_train, y_train) 
plt.show()

##------------Fin Task 1.3------------

##------------Task 1.4------------
CrossVal = []

salto = len(x_train)//5 

for i in range(0, 5):    
    x_train_split = x_train[i*salto:(i+1)*salto]
    y_train_split = y_train[i*salto:(i+1)*salto]
    CrossVal.append([x_train_split, y_train_split])
    
CrossVal = np.array(CrossVal)

def Matriz_CrossVal(index, CrossVal):
    x = []
    y = []
    for i in  range(len(CrossVal)):
        if i != index:
            x.extend(CrossVal[i][0])
            y.extend(CrossVal[i][1])
        
    return np.array(x), np.array(y)

polynomial_degree = 3

Error_pol = [0,0,0]

for pol in range(1, polynomial_degree + 1):
    
    for i in range(0, 5):
        
        x_train_split, y_train_split = Matriz_CrossVal(i, CrossVal)
        
        X, w, error = gradient_descent(pol,x_train_split,y_train_split)
        
        Error_pol[pol - 1] += error
        
    Error_pol[pol - 1] = Error_pol[pol - 1]/5
    

print(Error_pol)
print("El polinomio de grado " + str(Error_pol.index(min(Error_pol)) + 1) + " es el que mejor se ajusta al modelo con un error de " + str(min(Error_pol)))


best_pol = Error_pol.index(min(Error_pol)) + 1
X, w, error = gradient_descent(best_pol,x_test,y_test)
y_pred = (X @ w).transpose()

from sklearn.metrics import r2_score 

normalY = y_test.tolist()
normalY_pred = y_pred[0].tolist()

print("El valor de r^2 es de ",r2_score(y_test, y_pred[0]))
##------------Fin Task 1.4------------


##------------Task 1.5------------
"""
El polinomio que más se ajusto al modelo fue el de grado 1,
esto se debe a que los pies cuadrados de terreno vrs el precio
suelen tener una relación lineal, pero debido a que existen otras varaibles
que afectan el precio como lo es la ubicación, los comercios cercanos.
Podemos tener dos casas con la misma cantidad de pies cuadrados de terreno
con dos precios diferentes. Apesar de eso el polinomio de grado 1 permitió 
que tuvieramos un 0.48 de r^2 en nuestras predicciones, 
lo cual es un valor aceptable. En el gráfico se puede observar que la 
distribución de los datos no es lineal, lo cual apoya el argumento 
previo y también da una evidencia visual del valor de r^2.

"""