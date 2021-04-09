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

    trainingData = normalize(trainingData)
    testingData = normalize(testingData)
    W = training(trainingData)
    confMatrix = confusionMatrixCalc(W, testingData)



## Maybe use this for testing ###
def normalize(data):
    tempFeatures = data[:, 0:-2]
    tempClass = data[:,-2:]
    tempFeatures = tempFeatures/tempFeatures.max(axis=0)
    data = np.append(tempFeatures, tempClass, axis=1)
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
    #print("W shape = ", W.shape)
    tk_temp = np.zeros((nClasses, 1))
    #print(tk_temp)
    gk = np.zeros((nClasses))
    gk[0] = 1
    MSE = np.zeros(nIterations)
    mselist = []
    for i in range(nIterations):
        G_W_MSE = 0 ## Gradient of W_MSE
        testcounter = 0

        for xk in trainingData: ## Iterating through each single input

            #xk = np.insert(xk, -1, 0)
            #xk = np.append(xk, 1) ## Adding a one
            #print(xk[:-1])
            zk = np.matmul(W,(xk[:-1]))[np.newaxis].T

            gk = sigmoid(zk)

            ## Updating target vector
            tk_temp *= 0
            tk_temp[int(xk[-1]),:] = 1
            tk = tk_temp

            # Finding gradients for MSE calculation
            G_gk_MSE = gk-tk
            G_zk_g = np.multiply(gk, (1-gk))
            G_W_zk = xk[:-1].reshape(1,nFeatures+1) ### la til +1 her


            G_W_MSE += np.matmul(np.multiply(G_gk_MSE, (1-gk)), G_W_zk) ## Eq 22


            MSE[i] += 0.5* np.matmul((gk-tk).T,(gk-tk))


        # Moving W in opposite direction of the gradient
        W -= alpha*G_W_MSE

    #plt.plot(MSE)
    #plt.show()
    return W



def confusionMatrixCalc(W, testingData):

    confusionMatrix = np.zeros((nClasses, nClasses), dtype='float')

    for i in range(len(testingData)):
        classPrediction = int(np.argmax(np.matmul(W, testingData[i,0:5])))
        print("test", testingData[i,0:4])
        classActual = int(testingData[i, -1])
        confusionMatrix[classPrediction, classActual] += 1
    print(confusionMatrix)
    ## Plotting
    plt.imshow(confusionMatrix, cmap='hot', interpolation='nearest')
    plt.show()


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
    #print("before", trainingData[0:])

    ## Adding column of ones due to size differences in classification algorithm
    testData = np.insert(testData, -1, np.ones(testData.shape[0]), axis = 1)
    trainingData = np.insert(trainingData, -1, np.ones(trainingData.shape[0]), axis = 1)
    #print("now", trainingData)
    #print("class = ", testData[20,-1])
    #quit()


    return trainingData, testData



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
