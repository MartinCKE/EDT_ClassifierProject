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


def main():
    print("Loading iris dataset...")
    rawData = loadData()

    ## Scatter plot
    print("Plotting scatterplot of iris data...")
<<<<<<< HEAD
    scatterPlot(scatterData)

    ### SLETT ETTERPÅ
    print("Plotting histograms of types and features...")
    plotHistograms(data)

=======
    scatterPlot(rawData)
>>>>>>> 91c1087d86f075ea18025a075a6f95cf8cb1905c

    ### Normalizing data ###
    print("Normalizing data...")
    data = normalize(rawData)

    #### Training and classifying ####

    ## Task 1 a-c ##
    nTraining = 30
    nIterations = 4000
    trainingData, testingData = splitData(data, nTraining)
    #print("train", trainingData)
    #print("test", testingData)
    print("Training classifier with %d iterations and %d training " \
            "samples from each class." %(nIterations, nTraining))
    W = training(trainingData, nIterations)

    print("Plotting confusion matrix and finding error rate for training data...")
    confMatrix = confusionMatrixCalc(W, trainingData)
    plotConfusionMatrix(confMatrix, len(trainingData[0])-2)

    print("Plotting confusion matrix and finding error rate for testing data...")
    confMatrix2 = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix2, len(trainingData[0])-2)

    #Task 1 d-e ##
    #nTraining = 30
    #nIterations = 4000
    trainingData, testingData = splitData(data, nTraining, Flip=True)

    print("Training classifier with %d iterations and %d training " \
            "samples, using %d last samples." %(nIterations, nTraining, nTraining))
    W = training(trainingData, nIterations)

    print("Plotting confusion matrix and finding error rate for training data...")
    confMatrix = confusionMatrixCalc(W, trainingData)
    plotConfusionMatrix(confMatrix, len(trainingData[0])-2)

    print("Plotting confusion matrix and finding error rate for testing data...")
    confMatrix2 = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix2, len(trainingData[0])-2)

    ## Task 2 ##
    ### Plot histograms of dataset ###
<<<<<<< HEAD
    print("Plotting histograms of types and features...")
    plotHistograms(data)
=======
    print("Plotting histograms of species and features...")
    plotHistograms(rawData)
>>>>>>> 91c1087d86f075ea18025a075a6f95cf8cb1905c

    #### Removing features and repeating ####
    print("Removing feature with most overlap...")
    print("which is sepal width.")
    data = removeFeatures(data, [1])
    trainingData, testingData = splitData(data, nTraining)

    print("Training classifier with %d iterations and %d training " \
         "samples, using %d last samples." %(nIterations, nTraining, nTraining))

    W = training(trainingData, nIterations)

    print("Plotting confusion matrix and finding error rate for training data...")
    confMatrix = confusionMatrixCalc(W, trainingData)
    plotConfusionMatrix(confMatrix, len(trainingData[0])-2)

    print("Plotting confusion matrix and finding error rate for testing data...")
    confMatrix2 = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix2, len(trainingData[0])-2)

    print("Now removing one more feature (the one with most overlap)...")
    print("which is sepal length")
    data = removeFeatures(data, [0])
    trainingData, testingData = splitData(data, nTraining)

    print("Training classifier with %d iterations and %d training " \
         "samples, using %d last samples." %(nIterations, nTraining, nTraining))

    W = training(trainingData, nIterations)
    print("Weight matrix", W)

    print("Plotting confusion matrix and finding error rate for training data...")
    confMatrix = confusionMatrixCalc(W, trainingData)
    plotConfusionMatrix(confMatrix, len(trainingData[0])-2)

    print("Plotting confusion matrix and finding error rate for testing data...")
    confMatrix2 = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix2, len(trainingData[0])-2)


    print("Removing petal length, so only petal width is used is classifier")
    data = removeFeatures(data, [0])
    trainingData, testingData = splitData(data, nTraining)

    print("Training classifier with %d iterations and %d training " \
         "samples, using %d last samples." %(nIterations, nTraining, nTraining))

    W = training(trainingData, nIterations)
    print("Weight matrix", W)

    print("Plotting confusion matrix and finding error rate for training data...")
    confMatrix = confusionMatrixCalc(W, trainingData)
    plotConfusionMatrix(confMatrix, len(trainingData[0])-2)

    print("Plotting confusion matrix and finding error rate for testing data...")
    confMatrix2 = confusionMatrixCalc(W, testingData)
    plotConfusionMatrix(confMatrix2, len(trainingData[0])-2)


<<<<<<< HEAD
def scatterPlot(scatterData):
    col=['Sepal length [cm]','Sepal width [cm]','Petal length [cm]','Petal width [cm]','Type']
    iris = pd.DataFrame(scatterData, columns=col)
    iris['Sepal length [cm]'] = iris['Sepal length [cm]'].astype(float)
    iris['Sepal width [cm]'] = iris['Sepal width [cm]'].astype(float)
    iris['Petal length [cm]'] = iris['Petal length [cm]'].astype(float)
    iris['Petal width [cm]'] = iris['Petal width [cm]'].astype(float)



    iris_setosa = iris.loc[iris["Type"]=="Iris-setosa"]
    iris_versicolor = iris.loc[iris["Type"]=="Iris-versicolor"]
    iris_virginica = iris.loc[iris["Type"]=="Iris-virginica"]
    
=======
def scatterPlot(data):
    col=['Sepal length [cm]','Sepal width [cm]','Petal length [cm]','Petal width [cm]','Species']
    iris = pd.DataFrame(data, columns=col)

    ## Changing class ID back to text for plotting ##
    iris['Species'] = iris['Species'].replace([0.0], 'Iris Setosa')
    iris['Species'] = iris['Species'].replace([1.0], 'Iris Versicolor')
    iris['Species'] = iris['Species'].replace([2.0], 'Iris Virginica')
>>>>>>> 91c1087d86f075ea18025a075a6f95cf8cb1905c

    ## Make plot ##
    fig, axes = plt.subplots(2, 2, figsize=(8,6.5))
    fig.suptitle("Scatter plot of Iris types with different features")

<<<<<<< HEAD
    #sepal length
    slp = sns.scatterplot(ax=axes[0,0], data=iris, hue="Type", x="Sepal width [cm]", y="Sepal length [cm]", palette='copper', legend='auto')
    slp.legend(loc='upper right', prop={'size': 6})
    #sepal width
    swp = sns.scatterplot(ax=axes[0,1], data=iris, hue="Type", x="Petal width [cm]", y="Petal length [cm]", palette='copper', legend='auto')
    swp.legend(loc='upper left', prop={'size': 6})
    #petal length
    plp = sns.scatterplot(ax=axes[1,0], data=iris, hue="Type", x="Sepal width [cm]", y="Petal length [cm]", palette='copper', legend='auto')
    plp.legend(loc='upper right', prop={'size': 6})
    #petal width
    pwp = sns.scatterplot(ax=axes[1,1], data=iris, hue="Type", x="Petal width [cm]", y="Sepal length [cm]", palette='copper', legend='auto')
    pwp.legend(loc='upper left', prop={'size': 6})
=======
    #sepal length labels=['Iris setosa', 'Iris versicolor', 'Iris virginica'],
    slp = sns.scatterplot(ax=axes[0,0], data=iris, hue="Species", x="Sepal width [cm]", y="Sepal length [cm]", palette='copper')
    slp.legend(title='Species', loc='upper right',  prop={'size': 6})
    #sepal width
    swp = sns.scatterplot(ax=axes[0,1], data=iris, hue="Species", x="Petal width [cm]", y="Petal length [cm]", palette='copper')
    swp.legend(title='Species', loc='upper left', prop={'size': 6})
    #petal length
    plp = sns.scatterplot(ax=axes[1,0], data=iris, hue="Species", x="Sepal width [cm]", y="Petal length [cm]", palette='copper')
    plp.legend(title='Species', loc='upper right', prop={'size': 6})
    #petal width
    pwp = sns.scatterplot(ax=axes[1,1], data=iris, hue="Species", x="Petal width [cm]", y="Sepal length [cm]", palette='copper')
    pwp.legend(title='Species', loc='upper left', prop={'size': 6})
>>>>>>> 91c1087d86f075ea18025a075a6f95cf8cb1905c
    plt.show()


def findErrorRate(X):
    ''' Calculates error rate from confusion matrix '''
    errorRate = (1-np.sum(X.diagonal())/np.sum(X))*100
    return errorRate


def normalize(data):
    ''' Function which normalizes the feature values in dataset  '''
    tempFeatures = data[:, :-1]
    tempClass = data[:,-1:]
    tempFeatures = tempFeatures/tempFeatures.max(axis=0)
    data = np.append(tempFeatures, tempClass, axis=1)
    return data


def training(trainingData, nIterations, alpha = 0.04):

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
        t = 0
        for xk in trainingData: ## Iterating through each sample

            zk = np.matmul(W,(xk[:-1]))[np.newaxis].T
            #print("zk", zk)
            gk = sigmoid(zk)
            #print("gk", gk)
            t+=1
            #if t>=20:
            #    quit()

            ## Updating target vector
            tk_temp *= 0
            tk_temp[int(xk[-1]),:] = 1
            tk = tk_temp

            # Finding gradients for MSE calculation
            G_gk_MSE = gk-tk
            G_zk_g = np.multiply(gk, (1-gk))
            G_W_zk = xk[:-1].reshape(1,nFeatures+1)


            G_W_MSE += np.matmul(np.multiply(G_gk_MSE, (1-gk)), G_W_zk) ## Eq 22 from compendium


            MSE[i] += 0.5* np.matmul((gk-tk).T,(gk-tk)) ## Eq 19 from compendium


        # Moving W in opposite direction of the gradient
        W -= alpha*G_W_MSE


    plt.figure()
    plt.title("MSE converging over %d iterations "\
              "when %d features are used" %(nIterations, nFeatures))
    plt.plot(MSE)

    # textbox with final MSE converging value
    mseval = MSE[-1]
    textstr = ('MSE final value, %.2f' %(mseval))
    textBox = dict(boxstyle='round', facecolor='white', alpha=0.5) 
    plt.annotate('\n MSE converging value: %.2f \n' %mseval, xy=(0.56, 0.84), xycoords='axes fraction', bbox=textBox)
    plt.show()

    print("Weight matrix = ", W)
    return W



def confusionMatrixCalc(W, data):
    ''' Function which calculates the confusion matrix
        from trained classifier weight matrix and testing data.
    '''

    confusionMatrix = np.zeros((nClasses, nClasses), dtype='int')

    for i in range(len(data)):
        ### Predicting class by using weight matrix
        classPrediction = int(np.argmax(np.matmul(W, data[i,0:data.shape[1]-1])))
        ### Retreiving actual class
        classActual = int(data[i, -1])
        ### Adding prediction to confusion matrix
        confusionMatrix[classPrediction, classActual] += 1

    print("Confusion matrix \n")
    print(confusionMatrix)
    return confusionMatrix


def plotConfusionMatrix(confusionMatrix, nFeatures):
    ''' Function which plots confusion matrix as heat map with
        calculated error rate.
    '''
    ## Plotting
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(confusionMatrix, cmap='copper')
    ax.set_xticks(np.arange(0, nClasses))
    ax.set_yticks(np.arange(0, nClasses))
    ax.set_xticklabels(classLabels)
    ax.set_yticklabels(classLabels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ### Plotting error rate within heat map ###
    errorRate = findErrorRate(confusionMatrix)
    print("errorRate = ", errorRate)
    textstr = ('Error rate = %.1f %%\n nTraining = 30 \n nFeatures = %d' %(errorRate, nFeatures))
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
    plt.show()


def removeFeatures(data, featuresToRemove):
    ''' Funtion which removes feature(s) based on input int or array, where
        Sepal length = 0
        Sepal width = 1
        Petal length = 2
        Petal width = 3
    '''
    return np.delete(data, featuresToRemove, axis=1)


def plotHistograms(data):
    ''' Function for plotting histogram of iris dataset with its classes by
        different features.
    '''
    #Parse iris-data and histogramplot datasets in types with features format
    col=['Sepal length [cm]','Sepal width [cm]','Petal length [cm]','Petal width [cm]','Types']
    iris = pd.DataFrame(data, columns=col)

    iris_setosa = iris.loc[iris["Types"]=="0.0"]
    iris_versicolor = iris.loc[iris["Types"]=="1.0"]
    iris_virginica = iris.loc[iris["Types"]=="2.0"]

    fig, axes = plt.subplots(2, 2, figsize=(8,6.5))
<<<<<<< HEAD
    fig.suptitle("Distribution of Iris types based on features")
=======
    fig.suptitle("Histogram of Iris types based on features")
>>>>>>> 91c1087d86f075ea18025a075a6f95cf8cb1905c
    #sepal length                   labels=['Iris setosa', 'Iris versicolor', 'Iris virginica']
    slp = sns.histplot(ax=axes[0,0], data=iris, hue="Types", x="Sepal length [cm]", kde=True, binwidth=0.1, palette='copper')
    slp.legend(title='Types', loc='upper right', labels=['Virginica', 'Versicolor', 'Setosa'])
    #sepal width
    swp = sns.histplot(ax=axes[0,1], data=iris, hue="Types", x="Sepal width [cm]", kde=True, binwidth=0.1, palette='copper')
    swp.legend(title='Types', loc='upper right', labels=['Virginica', 'Versicolor', 'Setosa'])
    #petal length
    plp = sns.histplot(ax=axes[1,0], data=iris, hue="Types", x="Petal length [cm]", kde=True, binwidth=0.1, palette='copper')
    plp.legend(title='Types', loc='upper right', labels=['Virginica', 'Versicolor', 'Setosa'])
    #petal width
    pwp = sns.histplot(ax=axes[1,1], data=iris, hue="Types", x="Petal width [cm]", kde=True, binwidth=0.1, palette='copper')
    pwp.legend(title='Types', loc='upper right', labels=['Virginica', 'Versicolor', 'Setosa'])
    plt.show()

def sigmoid(x):
    ''' Calculating sigmoid function '''
    return np.array(1 / (1 + np.exp(-x)))

def splitData(data, nTraining, Flip = False):
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

        if Flip:
            trainingData[(i*nTraining):((i+1)*nTraining),:] = classNdata[nTest:,:]
            testData[(i*nTest):((i+1)*nTest),:] = classNdata[nTraining:, :]

        else:
            trainingData[(i*nTraining):((i+1)*nTraining),:] = classNdata[:nTraining,:]
            testData[(i*nTest):((i+1)*nTest),:] = classNdata[nTraining:, :]

    ## Adding column of ones due to size differences in classification algorithm
    testData = np.insert(testData, -1, np.ones(testData.shape[0]), axis = 1)
    trainingData = np.insert(trainingData, -1, np.ones(trainingData.shape[0]), axis = 1)

    return trainingData, testData



def loadData():
    ''' Function for reading data from file and assigning class ID '''

    ### Reading from file ###
    rawData = np.genfromtxt(irisDataLoc, dtype = str, delimiter=',')

    ### Assigning class ID instead of informative string ###
    for i, val in enumerate(rawData):
        classID = 0 if val[-1] == 'Iris-setosa' else 1 if val[-1] == 'Iris-versicolor' else 2
        rawData[i][4] = classID

    ### Converting from string to float
    data = rawData.astype(float)
    return data

if __name__ == '__main__':
    main()
