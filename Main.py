''' Project '''

import math
import numpy as np
import matplotlib.pyplot as plt
import os


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

irisDataLoc = os.path.join(__location__, 'Data/Iris_TTT4275/iris.data')

data = np.genfromtxt(irisDataLoc, dtype=None, delimiter=',', usecols=[0,1,2,3])
print(data)

plt.matshow(data)
plt.ylabel('irisData')
plt.show()