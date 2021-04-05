''' Project '''

import math
import numpy as np
import matplotlib.pyplot as plt
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

irisDataLoc = os.path.join(__location__, 'Data/Iris_TTT4275/iris.data')
#Class1Loc = os.path.join(__location__, 'Data/Iris_TTT4275/class_1')
#Class2Loc = os.path.join(__location__, 'Data/Iris_TTT4275/class_2')
#Class3Loc =os.path.join(__location__, 'Data/Iris_TTT4275/class_3')


nSamplesPerClass = 50
nClasses = 3
nFeatures = 4



def main():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    #data = np.genfromtxt(irisDataLoc, dtype=None, delimiter=',', usecols=[0,1,2,3])

    #NClasses = 3
    classLabels = ['Setosa','Versicolor','Virginica']

    data = loadData()
    trainingData, testingData = splitData(data, 30)
    for i in trainingData:
        print("okei", i)
    trainingData = normalize(trainingData)
    testingData = normalize(testingData)
    training(trainingData)

    #load_data()

    #training(t)

    #GetConfusionMatrix(t)


## Maybe use this for testing ###
def normalize(data):
    temp = data
    temp = temp/temp.max(axis=0)
    data = temp

    return data


def training(trainingData, nIterations = 1000, alpha = 0.05):

    ''' Training algorithm

        Variables:
            - G = Gradient
            - MSE = Mean Square Error
            - g_W_MSE = MSE of gradient to output vector
            - xk = Current input vector
            - zk = W*xk from compendium
            - gk = sigmoid of
            - tk = target vector containing class ID for current input vector

    '''
    W = np.zeros((nClasses, nFeatures+1))
    print("W shape = ", W.shape)
    tk_temp = np.zeros((nClasses, 1))
    print(tk_temp)
    gk = np.zeros((nClasses))
    gk[0] = 1
    MSE = np.zeros(nIterations)
    mselist = []
    for i in range(nIterations):
    #    MSE = 0 ## Mean Square Error
        G_W_MSE = 0 ## Gradient of W_MSE
        testcounter = 0
        for xk in trainingData: ## Iterating through each single input


            xk = np.insert(xk, -1, 0) ## Adding a one
            zk = np.matmul(W,(xk[:-1]))[np.newaxis].T
            gk = sigmoid(zk)

            tk_temp *= 0
            tk_temp[int(xk[-1]),:] = 1
            tk = tk_temp

            G_gk_MSE = gk-tk
            G_zk_g = np.multiply(gk, (1-gk))
            G_W_zk = xk[:-1].reshape(1,nFeatures+1) ### la til +1 her

            G_W_MSE += np.matmul(np.multiply(G_gk_MSE, (1-gk)), G_W_zk) ## Eq 22



            MSE[i] += 0.5* np.matmul((gk-tk).T,(gk-tk))
            #MSEtest += 0.5* np.matmul((gk-tk).T,(gk-tk))
            testcounter+=1
            if testcounter > 20:
                pass#break


        W -= alpha*G_W_MSE

    plt.plot(MSE)
    plt.show()



def GetConfusionMatrix(t):
    pass

def plotHistograms(X):
    pass

def sigmoid(x):
    ''' Simple function for calculating sigmoid '''
    return np.array(1 / (1 + np.exp(-x)))

def splitData(data, nTraining):
    '''
        Function for slitting data in to training and testing arrays
        Returns two numpy arrays (test and training) containing feature columns
        w/ class ID
    '''

    ### Constants

    nTest = nSamplesPerClass-nTraining

    ### Preallocate arrays
    trainingData = np.zeros((nTraining*nClasses, nFeatures+1))
    #print(trainingData.shape)
    testData = np.zeros((nTest*nClasses, nFeatures+1))
    #print(testData.shape)

    ### Iterating over classes and splitting data
    for i in range(nClasses):
        classNdata = data[(i*nSamplesPerClass):((i+1)*nSamplesPerClass), :] ## Gets the 50 values for each class
        trainingData[(i*nTraining):((i+1)*nTraining),:] = classNdata[:nTraining,:]
        testData[(i*nTest):((i+1)*nTest),:] = classNdata[nTraining:, :]

    return trainingData, testData
#
#


def loadData():
    ''' Function for reading data from file and assigning class ID '''

    ## Reading from files
    rawData = np.genfromtxt(irisDataLoc, dtype = str, delimiter=',',)

    ### Assigning class ID instead of informative string
    for i, val in enumerate(rawData):
        classID = 0 if val[-1] == 'Iris-setosa' else 1 if val[-1] == 'Iris-versicolor' else 2
        rawData[i][4] = classID

    data = rawData.astype(np.float) ## Converting from string to float
    return data

if __name__ == '__main__':
    main()
