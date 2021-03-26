''' Project '''

import math
import numpy as np
import matplotlib.pyplot as plt
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
irisDataLoc = os.path.join(__location__, 'Data/Iris_TTT4275/iris.data')
Class1Loc = os.path.join(__location__, 'Data/Iris_TTT4275/class_1')
Class2Loc = os.path.join(__location__, 'Data/Iris_TTT4275/class_2')
Class3Loc =os.path.join(__location__, 'Data/Iris_TTT4275/class_3')

def main():

    data = np.genfromtxt(irisDataLoc, dtype=None, delimiter=',', usecols=[0,1,2,3])
    #print(data)

    NClasses = 3
    Classes = ['Setosa','Versicolor','Virginica']

    readFiles(30)

def readFiles(NSubjects):
    ## Reading from files
    Class1_Data = open(Class1Loc, 'r').read().split('\n')
    Class2_Data = open(Class2Loc, 'r').read().split('\n')
    Class3_Data = open(Class3Loc, 'r').read().split('\n')
    #Splitting into training and testing
    T1 = Class1_Data[0:NSubjects]
    X1 = Class1_Data[NSubjects:]
    T2 = Class2_Data[0:NSubjects]
    X2 = Class2_Data[NSubjects:]
    T3 = Class3_Data[0:NSubjects]
    X3 = Class3_Data[NSubjects:]

    print(T1)
    print(X1)


if __name__ == '__main__':
    main()
