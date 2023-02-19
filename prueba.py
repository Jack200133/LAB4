import matplotlib.pyplot as plt
from scipy import stats

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
mymodel = np.poly1d(np.polyfit(x_train, y_train, 2))

myline = np.linspace(1, 14000)

plt.scatter(x_train, y_train)
plt.plot(myline, mymodel(myline))
plt.show()