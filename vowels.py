import numpy as np
import os
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sn


'''
Filenames:
character 1:     m=man, w=woman, b=boy, g=girl
characters 2-3:  talker number
characters 4-5:  vowel (ae="had", ah="hod", aw="hawed", eh="head", er="heard",
                        ei="haid", ih="hid", iy="heed", oa=/o/ as in "boat",
                        oo="hood", uh="hud", uw="who'd")

col1:  filename
col2:  duration in msec
col3:  f0 at "steady state"
col4:  F1 at "steady state"
col5:  F2 at "steady state"
col6:  F3 at "steady state"
col7:  F4 at "steady state"
col8:  F1 at 20% of vowel duration
col9:  F2 at 20% of vowel duration
col10: F3 at 20% of vowel duration
col11: F1 at 50% of vowel duration
col12: F2 at 50% of vowel duration
col13: F3 at 50% of vowel duration
col14: F1 at 80% of vowel duration
col15: F2 at 80% of vowel duration
col16: F3 at 80% of vowel duration

'''

vowels = ['ae','ah','aw','eh','er','ei','ih','iy','oa','oo','uh','uw']
talkers = ['m', 'w', 'b', 'g']

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


vowelDataLoc = os.path.join(__location__, 'Data/Wowels/vowdata_nohead.dat')



def main():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    global features
    nTraining = 7
    features=range(3,6)
    M = 2

    types, data = loadData(features)

    vowelCol = pd.Series([col1[3:] for col1 in types])

    df = pd.DataFrame(data, index=vowelCol)
    trainingData, testingData = splitData(df, nTraining)

    ### Task 1 ###
    #vowelModels = singleGMTraining(trainingData, True)
    ##predictions, actualVowels = singleGMTesting(vowelModels, testingData)
    #confMatrix = confusionMatrixCalc(predictions, actualVowels)
    #plotConfusionMatrix(confMatrix)
    #print(confMatrix)

    ### Task 2 ###
    GaussianMixtureModels = GMMTraining(trainingData, M)
    predictions, actualVowels = GMMTesting(GaussianMixtureModels, testingData, M)
    confMatrix = confusionMatrixCalc(predictions, actualVowels)
    plotConfusionMatrix(confMatrix)
    print(confMatrix)

def findErrorRate(X):
    ''' Calculates error rate from confusion matrix '''
    errorRate = (1-np.sum(X.diagonal())/np.sum(X))*100
    return errorRate

def getMeanAndCovariance(df, features, vowel, diag=False):
    '''
        Function which finds mean and covariance of given vowel and
        features to use.
    '''
    vowelDataFrame = df.loc[vowel]
    mean = vowelDataFrame.mean(axis=0).values
    cov = vowelDataFrame.cov().values

    if diag:
       cov = np.diag(np.diag(cov))

    return mean, cov

def confusionMatrixCalc(predictions, actualVowels, diag=False):

    confMatrix = np.zeros((len(vowels), len(vowels)))

    for i in range(len(predictions)):
        vowelIndex = vowels.index(actualVowels[i])
        confMatrix[vowelIndex][(predictions[i])] += 1

    return confMatrix

def plotConfusionMatrix(confusionMatrix):
    ''' Function which plots confusion matrix as heat map with
        calculated error rate.
    '''
    ## Plotting
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(confusionMatrix, cmap='copper')
    ax.set_xticks(np.arange(0, len(vowels)))
    ax.set_yticks(np.arange(0, len(vowels)))
    ax.set_xticklabels(vowels)
    ax.set_yticklabels(vowels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ### Plotting error rate within heat map ###
    errorRate = findErrorRate(confusionMatrix)
    print("errorRate = ", errorRate)
    #textstr = ('Error rate = %.1f %%\n nTraining = 30' %(errorRate))
    textstr = ('Error rate = %.1f%%' %(errorRate))
    textBox = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.72, 0.935, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=textBox)

    # Creating text annotations in cells
    for i in range(0, len(vowels)):
        for j in range(0, len(vowels)):
            text = ax.text(j, i, confusionMatrix[i,j],
                           ha="center", va="center", color="w")

    ax.set_title("Heatmap visualizing confusion matrix")
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()


def GMMTesting(GaussianMixtureModels, testingData, M):

    probabilities = np.zeros((len(vowels), len(testingData)))

    for i, vowel in enumerate(vowels):

        GaussianMixtureModel = GaussianMixtureModels[i]

        for j in range(M):
            curve = multivariate_normal(mean=GaussianMixtureModel.means_[j],
                                        cov=GaussianMixtureModel.covariances_[j], allow_singular=True)
            probabilities[i] += GaussianMixtureModel.weights_[j] * curve.pdf(testingData.values)

    print(probabilities)
    predictions = np.argmax(probabilities, axis=0)
    actualVowels = testingData.index.values
    return predictions, actualVowels


def GMMTraining(trainingData, M):

        GaussianMixtureModels = []


        for vowel in vowels:
            #training_values = train_data.loc[vowel].values[:, :-2]
            trainingVowelData = trainingData.loc[vowel].values

            gmm = GMM(n_components=M, covariance_type='diag') #,reg_covar=1e-4, random_state=0)
            gmm.fit(trainingVowelData)
            GaussianMixtureModels.append(gmm)

        return GaussianMixtureModels

def singleGMTraining(trainingData, diag=False):

    vowelModels = list()

    for vowel in vowels:
        mean, covariance = getMeanAndCovariance(trainingData, features, vowel, diag)

        multiVariateModel = multivariate_normal(mean = mean, cov=covariance)
        vowelModels.append(multiVariateModel)

    return vowelModels

def singleGMTesting(vowelModels, testingData):

    probabilities = np.zeros((len(vowels), len(testingData)))

    for i, vowel in enumerate(vowels):
        i_vowel = vowelModels[i]
        probabilities[i] = i_vowel.pdf(testingData.values)
    print(probabilities)

    predictions = np.argmax(probabilities, axis=0)

    actualVowels = testingData.index.values

    return predictions, actualVowels

def splitData(df, nTraining):
    trainingdf = pd.DataFrame()
    testingdf = pd.DataFrame()

    trainingdf = pd.concat([trainingdf.append(df.loc[vowel][:nTraining]) for vowel in vowels])
    testingdf = pd.concat([testingdf.append(df.loc[vowel][nTraining:]) for vowel in vowels])

    return trainingdf, testingdf


def loadData(features):
    '''
        features = which features to use
    '''
    rawData = np.genfromtxt(vowelDataLoc, dtype = str, delimiter=',',)
    types = list()
    data = np.zeros((len(rawData), len(features)), dtype='int')

    for i, line in enumerate(rawData):
        line = line.split()
        types.append(line[0])
        for index, val in enumerate(features):
            data[i, index] = int(line[val])

    types = np.array(types)
    return types, data


if __name__ == '__main__':
    main()
