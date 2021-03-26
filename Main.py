''' Project '''

import math
import numpy
import matplotlib as plt
import os


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

irisDataLoc = os.path.join(__location__, 'Data/Iris_TTT4275/iris.data')

with open(irisDataLoc) as file: ##
    for line in file:
        print(line)
        for word in line:
            print(word)
