#Name: Peter Bui#Class: Machine Learning 445 - Winter Term
#Purpose: Trains 10 perceptrons that, as a group, learn to
#identify handwritten digits in the MNIST dataset

print("Program Start")

#Packages
#Useful math functions, np as shorthand
import numpy as np
#Useful to keep track of execution time
import time as time
start_time = time.time()
#Useful for plotting data... Doesnt seem to work
#import matplotlib as plt

#Load test data, these will be converted from csv to numpy arrays
#These arrays would basically look like [ [...], [...] ]
#For this project, the datasets are 60000 x 785
from numpy import genfromtxt

#Should be 60000 examples
trainingData = genfromtxt('mnist_train.csv', dtype='int', delimiter=',')
#Should be 10000 
testingData = genfromtxt('mnist_test.csv', dtype='int', delimiter=',')
print("Dataset loading: --- %s seconds ---" % (time.time() - start_time))

#Prepare the confusion matrix here to track the amount of times a preceptron guesses a certain number
#This is just a 10x10 grid that increases the value of each cell every time a guess is made there
confusionMatrix = np.zeros((10,10))
confusionMatrixTst = np.zeros((10,10))

#This array will keep track of the accuracy for each epoch
#We will append an accuracy rating after each epoch
accuracyFinal = []
accuracyFinalTst = []

print('Training data length: ',len(trainingData))
print('Testing data length: ',len(testingData))
#Preprocessing the data:

#for the datasets. We also need to randomly shuffle the data each time we conduct an experiment
#These two lines shuffles the data along one axis, so numbers in each row stays the same, but the
#rows will be in a different order
np.random.shuffle(trainingData)
np.random.shuffle(testingData)

#Get the target labels
targetTrainingLabels = []
for L in range(len(trainingData)):
    targetTrainingLabels.append(trainingData[L][0])

targetTestingLabels = []
for L in range(len(testingData)):
    targetTestingLabels.append(testingData[L][0])


#Scale each data value in these datasets to be between 0 and 1
#This can be done by dividing each of them by 255, which is the max of these values

trainingData = np.true_divide(trainingData[:,1:len(trainingData[0])], 255)
testingData = np.true_divide(testingData[:,1:len(testingData[0])], 255)

#Since we saved the target labels, we can replace the first column of our dataset with
#Our bias input, which should always be 1
bias = np.ones((len(trainingData),1))
biasTst = np.ones((len(testingData),1))
trainingData = np.concatenate((bias,trainingData),axis=1)
testingData = np.concatenate((biasTst,testingData),axis=1)

#Convert the labels into target vectors to compare later
targetTrainingData = []
for x in range(len(trainingData)):
    if targetTrainingLabels[x] == 0:
        targetTrainingData.append([1,0,0,0,0,0,0,0,0,0])
    elif targetTrainingLabels[x] == 1:
        targetTrainingData.append([0,1,0,0,0,0,0,0,0,0])
    elif targetTrainingLabels[x] == 2:
        targetTrainingData.append([0,0,1,0,0,0,0,0,0,0])
    elif targetTrainingLabels[x] == 3:
        targetTrainingData.append([0,0,0,1,0,0,0,0,0,0])
    elif targetTrainingLabels[x] == 4:
        targetTrainingData.append([0,0,0,0,1,0,0,0,0,0])
    elif targetTrainingLabels[x] == 5:
        targetTrainingData.append([0,0,0,0,0,1,0,0,0,0])
    elif targetTrainingLabels[x] == 6:
        targetTrainingData.append([0,0,0,0,0,0,1,0,0,0])
    elif targetTrainingLabels[x] == 7:
        targetTrainingData.append([0,0,0,0,0,0,0,1,0,0])
    elif targetTrainingLabels[x] == 8:
        targetTrainingData.append([0,0,0,0,0,0,0,0,1,0])
    elif targetTrainingLabels[x] == 9:
        targetTrainingData.append([0,0,0,0,0,0,0,0,0,1])

targetTestingData = []
for x in range(len(testingData)):
    if targetTestingLabels[x] == 0:
        targetTestingData.append([1,0,0,0,0,0,0,0,0,0])
    elif targetTestingLabels[x] == 1:
        targetTestingData.append([0,1,0,0,0,0,0,0,0,0])
    elif targetTestingLabels[x] == 2:
        targetTestingData.append([0,0,1,0,0,0,0,0,0,0])
    elif targetTestingLabels[x] == 3:
        targetTestingData.append([0,0,0,1,0,0,0,0,0,0])
    elif targetTestingLabels[x] == 4:
        targetTestingData.append([0,0,0,0,1,0,0,0,0,0])
    elif targetTestingLabels[x] == 5:
        targetTestingData.append([0,0,0,0,0,1,0,0,0,0])
    elif targetTestingLabels[x] == 6:
        targetTestingData.append([0,0,0,0,0,0,1,0,0,0])
    elif targetTestingLabels[x] == 7:
        targetTestingData.append([0,0,0,0,0,0,0,1,0,0])
    elif targetTestingLabels[x] == 8:
        targetTestingData.append([0,0,0,0,0,0,0,0,1,0])
    elif targetTestingLabels[x] == 9:
        targetTestingData.append([0,0,0,0,0,0,0,0,0,1])
        
#Learnings rates to be tested are:
#CHANGE LEARNING RATE HERE
learningRate = 0.1
#Experiment 1: 0.001
#Experiment 2: 0.01
#Experiment 3: 0.1

#Make sure to get random initial weights [-.05,.05] for each test
#Uniform distribution should give numbers 0 to 1, then we divide by .05 to "normalize" the values
#There are 28x28+1 weights per y calculation -> 785, one for each x input for a label
###INITIAL WEIGHTS
weights = np.random.uniform(low=-0.05, high=0.05, size=785*10).reshape(10,785)
weightsTst = np.random.uniform(low=-0.05, high=0.05, size=785*10).reshape(10,785)

###START EPOCH
#There will be 10 Perceptrons, we can represent them as a an array of outputs
trainingPerceptrons = np.zeros((10,len(trainingData)))
testingPerceptrons = np.zeros((10,len(testingData)))

#There will be 50 Epochs
for e in range(50):

    #This array will keep track of each example's prediction
    predictions = []
    predictionsTst = []
    
    #PERCEPTRON LEARNING RULE
    #Now we need to have each perceptron calculate a new weight if applicable

    #Formula: w = w + n(t-y)x

    #k is one of the dataset's examples,
    for k in range(len(trainingPerceptrons[0])):
        #Select next training example and run the perceptrons with input x and weights w
        #This function will multiply each weight for all perceptrons with the respective input data
        #And give us 1x10 y outputs, one for each perceptron
        wxOutputs = np.dot(weights,trainingData[k])
        yOutputs = np.dot(weights,trainingData[k])

        #Save the highest wxOutputs to select as a prediction
        predictions.append(np.argmax(wxOutputs))

        #Determine whether or not the y output is 0 or 1
        #1 if the wx is > 0 and 0 otherwise
        np.place(yOutputs,yOutputs>0,[1])
        np.place(yOutputs,yOutputs<0,[0])

        #Subtract yOutputs from the target, then get the outer product of the current example's inputs
        #Next, multiply with the scalar learning rate
        #We should have a 10x785 matrix of n*(t-y)*x 
        ntminusytimesx = np.multiply(learningRate,np.outer(np.subtract(targetTrainingData[k],yOutputs),trainingData[k]))

        #With each of the 10x785 weights accounted for, we can simply add the two matrices
        #Positive numbers would increase the weight while negative ones will decrease it
        weights = np.add(weights,ntminusytimesx)

    for k in range(len(testingPerceptrons[0])):
        wxOutputsTst = np.dot(weightsTst,testingData[k])
        yOutputsTst = np.dot(weightsTst,testingData[k])

        predictionsTst.append(np.argmax(wxOutputsTst))

        np.place(yOutputsTst,yOutputsTst>0,[1])
        np.place(yOutputsTst,yOutputsTst<0,[0])

        ntminusytimesxTsT = np.multiply(learningRate,np.outer(np.subtract(targetTestingData[k],yOutputsTst),testingData[k]))

        weightsTst = np.add(weightsTst,ntminusytimesxTsT)

    
    print("Calculate n(t-y)x done: --- %s seconds ---" % (time.time() - start_time))

    #After weights have changed, it's time for the perceptrons to guess again, next epoch!

    print("Epoch ", e, " --- %s seconds ---" % (time.time() - start_time))

    ###PREDICTION SELECTION
    #Find the max in each of the 60000 examples of wxOutputs columns and put the pereptron number into an array
    #Then, convert the values to compare to target testing data later

    predictionsConverted = []
    predictionsConvertedTst = []

    for m in range(len(trainingPerceptrons[0])):
        if predictions[m] == 0:
            predictionsConverted.append([1,0,0,0,0,0,0,0,0,0])
        elif predictions[m] == 1:
            predictionsConverted.append([0,1,0,0,0,0,0,0,0,0])
        elif predictions[m] == 2:
            predictionsConverted.append([0,0,1,0,0,0,0,0,0,0])
        elif predictions[m] == 3:
            predictionsConverted.append([0,0,0,1,0,0,0,0,0,0])
        elif predictions[m] == 4:
            predictionsConverted.append([0,0,0,0,1,0,0,0,0,0])
        elif predictions[m] == 5:
            predictionsConverted.append([0,0,0,0,0,1,0,0,0,0])
        elif predictions[m] == 6:
            predictionsConverted.append([0,0,0,0,0,0,1,0,0,0])
        elif predictions[m] == 7:
            predictionsConverted.append([0,0,0,0,0,0,0,1,0,0])
        elif predictions[m] == 8:
            predictionsConverted.append([0,0,0,0,0,0,0,0,1,0])
        elif predictions[m] == 9:
            predictionsConverted.append([0,0,0,0,0,0,0,0,0,1])
            
    for m in range(len(testingPerceptrons[0])):
        if predictionsTst[m] == 0:
            predictionsConvertedTst.append([1,0,0,0,0,0,0,0,0,0])
        elif predictionsTst[m] == 1:
            predictionsConvertedTst.append([0,1,0,0,0,0,0,0,0,0])
        elif predictionsTst[m] == 2:
            predictionsConvertedTst.append([0,0,1,0,0,0,0,0,0,0])
        elif predictionsTst[m] == 3:
            predictionsConvertedTst.append([0,0,0,1,0,0,0,0,0,0])
        elif predictionsTst[m] == 4:
            predictionsConvertedTst.append([0,0,0,0,1,0,0,0,0,0])
        elif predictionsTst[m] == 5:
            predictionsConvertedTst.append([0,0,0,0,0,1,0,0,0,0])
        elif predictionsTst[m] == 6:
            predictionsConvertedTst.append([0,0,0,0,0,0,1,0,0,0])
        elif predictionsTst[m] == 7:
            predictionsConvertedTst.append([0,0,0,0,0,0,0,1,0,0])
        elif predictionsTst[m] == 8:
            predictionsConvertedTst.append([0,0,0,0,0,0,0,0,1,0])
        elif predictionsTst[m] == 9:
            predictionsConvertedTst.append([0,0,0,0,0,0,0,0,0,1])
        
    
    ###ACCURACY CALCULATION
    #Take the accuracy and divide it over the correct guesses to find out how often the perceptrons got a right guess
    correctGuesses = 0
    correctGuessesTst = 0
    
    for a in range(len(trainingData)):
    #Increase the confusion matrix cell for each respective guess vs target

        confusionMatrix[predictions[a]][targetTrainingLabels[a]] = confusionMatrix[predictions[a]][targetTrainingLabels[a]] + 1
    #Also increase the amount of correct guesses
        if np.array_equal(predictionsConverted[a],  targetTrainingData[a]):
            correctGuesses = correctGuesses + 1

    for a in range(len(testingData)):
        confusionMatrixTst[predictionsTst[a]][targetTestingLabels[a]] = confusionMatrixTst[predictionsTst[a]][targetTestingLabels[a]] + 1
        if np.array_equal(predictionsConvertedTst[a],  targetTestingData[a]):
            correctGuessesTst = correctGuessesTst + 1
            

    accuracyFinal.append(correctGuesses/len(trainingData))
    accuracyFinalTst.append(correctGuessesTst/len(testingData))


#After all epochs have passed, display the confusion matrix and the accuracies of each epoch
for acc in range(len(accuracyFinal)):
    print(accuracyFinal[acc])
for cm in range(len(confusionMatrix)):
    print(confusionMatrix[cm])

for acc in range(len(accuracyFinalTst)):
    print(accuracyFinalTst[acc])
for cm in range(len(confusionMatrixTst)):
    print(confusionMatrixTst[cm])
print("Total time: --- %s seconds ---" % (time.time() - start_time))
