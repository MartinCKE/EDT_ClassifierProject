''' Project '''

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import seaborn as sns

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

irisDataLoc = os.path.join(__location__, 'Data/Iris_TTT4275/iris.data')

nSamplesPerClass = 50
nClasses = 3
nFeatures = 4
classLabels = ['Setosa','Versicolor','Virginica']
features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
nTraining = 20
nIterations = 1000


def main():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    print("Loading iris dataset...")
    data = loadData()

    #Plot histogram from dataset
    plotHistograms(data)


    #### Training and classifying ####
    trainingData, testingData = splitData(data, nTraining)
    trainingData = normalize(trainingData)
    testingData = normalize(testingData)
    print("Training classifier with %d iterations and %d training " \
            "samples from each class." %(nIterations, nTraining))
    W = training(trainingData, nIterations)
    #print("MSE", MSE)
    #print("W", type(W))
    print("Plotting confusion matrix...")
    confMatrix = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix, nTraining)

    #### Removing features and repeating ####
    print("Removing features and repeating training")
    data = removeFeatures(data, [0,2,3])

    trainingData, testingData = splitData(data, nTraining)

    trainingData = normalize(trainingData)
    testingData = normalize(testingData)
    W = training(trainingData, nIterations)
    confMatrix = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix, nTraining)
    #plotHistograms(data, 0.1)
    plt.show()



def findErrorRate(confusionMatrix):
    ''' Calculates error rate from confusion matrix '''
    errorRate = (1-np.sum(confusionMatrix.diagonal())/np.sum(confusionMatrix))*100
    return errorRate


def normalize(data):
    ''' Function which normalizes the feature measures  '''
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
            - gk = sigmoid of zk
            - tk = target vector containing class ID for current input vector

    '''

    nFeatures = trainingData.shape[1]-2
    W = np.zeros((nClasses, nFeatures+1))
    tk_temp = np.zeros((nClasses, 1))
    gk = np.zeros((nClasses))
    MSE = np.zeros(nIterations)


    for i in range(nIterations):
        G_W_MSE = 0

        for xk in trainingData: ## Iterating through each sample

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


            G_W_MSE += np.matmul(np.multiply(G_gk_MSE, (1-gk)), G_W_zk) ## Eq 22 from compendium


            MSE[i] += 0.5* np.matmul((gk-tk).T,(gk-tk)) ## Eq 19 from compendium


        # Moving W in opposite direction of the gradient
        W -= alpha*G_W_MSE


    plt.figure()
    plt.title("MSE converging over %d iterations "\
              "when %d features are used" %(nIterations, nFeatures))
    plt.plot(MSE)
    plot = False
    plt.show()

    return W



def confusionMatrixCalc(W, testingData):
    ''' Function which calculates the confusion matrix
        from trained classifier weight matrix and testing data.
    '''
    print("W er", W)
    confusionMatrix = np.zeros((nClasses, nClasses), dtype='float')

    for i in range(len(testingData)):
        ### Predicting class by using weight matrix
        classPrediction = int(np.argmax(np.matmul(W, testingData[i,0:testingData.shape[1]-1])))
        ### Retreiving actual class
        classActual = int(testingData[i, -1])
        ### Adding prediction to confusion matrix
        confusionMatrix[classPrediction, classActual] += 1

    print(confusionMatrix)
    return confusionMatrix


def plotConfusionMatrix(confusionMatrix, nTraining):
    ''' Function which plots confusion matrix as heat map with
        calculated error rate.
    '''

    ## Plotting
    fig, ax = plt.subplots()
    im = ax.imshow(confusionMatrix, cmap='copper')
    ax.set_xticks(np.arange(0, nClasses))
    ax.set_yticks(np.arange(0, nClasses))
    ax.set_xticklabels(classLabels)
    ax.set_yticklabels(classLabels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    ### Plotting error rate within heat map ###
    errorRate = findErrorRate(confusionMatrix)
    textstr = ('Error rate = %.1f %%\n nTraining = %d' %(errorRate, nTraining))
    textBox = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.60, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=textBox)

    # Creating text annotations in cells
    for i in range(0, nClasses):
        for j in range(0, nClasses):
            text = ax.text(j, i, confusionMatrix[i,j],
                           ha="center", va="center", color="w")

    ax.set_title("Heatmap visualizing confusion matrix")
    fig.tight_layout()
    plt.colorbar(im)


def removeFeatures(data, featuresToRemove):
    ''' Funtion which removes feature(s) based on input int or array, where
        Sepal length = 0
        Sepal width = 1
        Petal length = 2
        Petal width = 3
    '''
    return np.delete(data, featuresToRemove, axis=1)



def plotHistograms(data):
    #Parse iris-data and histogramplot datasets in species with features format
    col=['Sepal length [cm]','Sepal width [cm]','Petal length [cm]','Petal width [cm]','Species']
    iris = pd.DataFrame(data, columns=col)

    iris_setosa = iris.loc[iris["Species"]=="0.0"]
    iris_versicolor = iris.loc[iris["Species"]=="1.0"]
    iris_virginica = iris.loc[iris["Species"]=="2.0"]

    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    fig.suptitle("Distribution of species based on features")
    #sepal length
    slp = sns.histplot(ax=axes[0,0], data=iris, hue="Species", x="Sepal length [cm]", kde=True)
    slp.legend(title='Species', loc='upper right', labels=['Iris setosa', 'Iris versicolor', 'Iris virginica'])
    #sepal width
    swp = sns.histplot(ax=axes[0,1], data=iris, hue="Species", x="Sepal width [cm]", kde=True)
    swp.legend(title='Species', loc='upper right', labels=['Iris setosa', 'Iris versicolor', 'Iris virginica'])
    #petal length
    plp = sns.histplot(ax=axes[1,0], data=iris, hue="Species", x="Petal length [cm]", kde=True)
    plp.legend(title='Species', loc='upper right', labels=['Iris setosa', 'Iris versicolor', 'Iris virginica'])
    #petal width
    pwp = sns.histplot(ax=axes[1,1], data=iris, hue="Species", x="Petal width [cm]", kde=True)
    pwp.legend(title='Species', loc='upper right', labels=['Iris setosa', 'Iris versicolor', 'Iris virginica'])


def sigmoid(x):
    ''' Calculating sigmoid function '''
    return np.array(1 / (1 + np.exp(-x)))

def splitData(data, nTraining):
    '''
        Function for slitting data in to training and testing arrays
        Returns two numpy arrays (test and training) containing feature columns
        w/ class ID
    '''

    ### Constants
    nFeatures = data.shape[1]-1
    nTest = nSamplesPerClass-nTraining

    ### Preallocate arrays
    trainingData = np.zeros((nTraining*nClasses, nFeatures+1))
    testData = np.zeros((nTest*nClasses, nFeatures+1))

    ### Iterating over classes and splitting data
    for i in range(nClasses):
        classNdata = data[(i*nSamplesPerClass):((i+1)*nSamplesPerClass), :] ## Gets the 50 values for each class
        trainingData[(i*nTraining):((i+1)*nTraining),:] = classNdata[:nTraining,:]
        testData[(i*nTest):((i+1)*nTest),:] = classNdata[nTraining:, :]

    ## Adding column of ones due to size differences in classification algorithm
    testData = np.insert(testData, -1, np.ones(testData.shape[0]), axis = 1)
    trainingData = np.insert(trainingData, -1, np.ones(trainingData.shape[0]), axis = 1)

    return trainingData, testData



def loadData():
    ''' Function for reading data from file and assigning class ID '''

    ### Reading from file ###
    rawData = np.genfromtxt(irisDataLoc, dtype = str, delimiter=',',)

    ### Assigning class ID instead of informative string ###
    for i, val in enumerate(rawData):
        classID = 0 if val[-1] == 'Iris-setosa' else 1 if val[-1] == 'Iris-versicolor' else 2
        rawData[i][4] = classID

    ### Converting from string to float
    data = rawData.astype(np.float)
    return data

if __name__ == '__main__':
    main()
