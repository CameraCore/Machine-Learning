#Name: Peter Bui
#Class: Machine Learning 445 - Winter Term
#Purpose: Trains 10 perceptrons that, as a group, learn to
#identify handwritten digits in the MNIST dataset
#This one uses a single hidden layer to increase its ability
#to perform better

print("Program Start")


#Packages
#Useful math functions, np as shorthand
import numpy as np
#Useful to keep track of execution time
import time as time
start_time = time.time()
np.set_printoptions(suppress=True)

#Load test data, these will be converted from csv to numpy arrays
#These arrays would basically look like [ [...], [...] ]
#For this project, the datasets are 60000 x 785
from numpy import genfromtxt

###PREPROCESSING###

#Should be 60000 examples
trainingData = genfromtxt('mnist_train.csv', dtype='int', delimiter=',')
#Should be 10000 
testingData = genfromtxt('mnist_test.csv', dtype='int', delimiter=',')
#UNCOMMENT WHEN READY TO USE REAL TEST DATA
print("Dataset loading: --- %s seconds ---" % (time.time() - start_time))

#Prepare the confusion matrix here to track the amount of times a preceptron guesses a certain number
#This is just a 10x10 grid that increases the value of each cell every time a guess is made there
confusionMatrix = np.zeros((10,10))

#This array will keep track of the accuracy for each epoch
#We will append an accuracy rating after each epoch
accuracyFinal = []
accuracyFinalTst = []

#LEARNING RATE - We aren't changing it for this assignment
#learningRate
lR = 0.1

#PROTOTYPE DATA
protoData = [[1,1,0],[1,1,0]]
#PROTOTYPE LABELS
protoLabels = [[0.9,0.1],[0.9,0.1]]

#for the datasets. We also need to randomly shuffle the data each time we conduct an experiment
#These two lines shuffles the data along one axis, so numbers in each row stays the same, but the
#rows will be in a different order
np.random.shuffle(trainingData)
np.random.shuffle(testingData)


###FOR EXPERIMENT 2
#Chop off a 1/2 or 1/4 of the training data


#trainingData = trainingData[:30000]
trainingData = trainingData[:15000]

print("Current training length: ", len(trainingData))




#Get the target labels by looking at the first column of the datasets
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
        targetTrainingData.append([.9,.1,.1,.1,.1,.1,.1,.1,.1,.1])
    elif targetTrainingLabels[x] == 1:
        targetTrainingData.append([.1,.9,.1,.1,.1,.1,.1,.1,.1,.1])
    elif targetTrainingLabels[x] == 2:
        targetTrainingData.append([.1,.1,.9,.1,.1,.1,.1,.1,.1,.1])
    elif targetTrainingLabels[x] == 3:
        targetTrainingData.append([.1,.1,.1,.9,.1,.1,.1,.1,.1,.1])
    elif targetTrainingLabels[x] == 4:
        targetTrainingData.append([.1,.1,.1,.1,.9,.1,.1,.1,.1,.1])
    elif targetTrainingLabels[x] == 5:
        targetTrainingData.append([.1,.1,.1,.1,.1,.9,.1,.1,.1,.1])
    elif targetTrainingLabels[x] == 6:
        targetTrainingData.append([.1,.1,.1,.1,.1,.1,.9,.1,.1,.1])
    elif targetTrainingLabels[x] == 7:
        targetTrainingData.append([.1,.1,.1,.1,.1,.1,.1,.9,.1,.1])
    elif targetTrainingLabels[x] == 8:
        targetTrainingData.append([.1,.1,.1,.1,.1,.1,.1,.1,.9,.1])
    elif targetTrainingLabels[x] == 9:
        targetTrainingData.append([.1,.1,.1,.1,.1,.1,.1,.1,.1,.9])
    

targetTestingData = []
for x in range(len(testingData)):
    if targetTestingLabels[x] == 0:
        targetTestingData.append([.9,.1,.1,.1,.1,.1,.1,.1,.1,.1])
    elif targetTestingLabels[x] == 1:
        targetTestingData.append([.1,.9,.1,.1,.1,.1,.1,.1,.1,.1])
    elif targetTestingLabels[x] == 2:
        targetTestingData.append([.1,.1,.9,.1,.1,.1,.1,.1,.1,.1])
    elif targetTestingLabels[x] == 3:
        targetTestingData.append([.1,.1,.1,.9,.1,.1,.1,.1,.1,.1])
    elif targetTestingLabels[x] == 4:
        targetTestingData.append([.1,.1,.1,.1,.9,.1,.1,.1,.1,.1])
    elif targetTestingLabels[x] == 5:
        targetTestingData.append([.1,.1,.1,.1,.1,.9,.1,.1,.1,.1])
    elif targetTestingLabels[x] == 6:
        targetTestingData.append([.1,.1,.1,.1,.1,.1,.9,.1,.1,.1])
    elif targetTestingLabels[x] == 7:
        targetTestingData.append([.1,.1,.1,.1,.1,.1,.1,.9,.1,.1])
    elif targetTestingLabels[x] == 8:
        targetTestingData.append([.1,.1,.1,.1,.1,.1,.1,.1,.9,.1])
    elif targetTestingLabels[x] == 9:
        targetTestingData.append([.1,.1,.1,.1,.1,.1,.1,.1,.1,.9])
    

##HIDDEN ACTIVATION UNITS
#VARY THE AMOUNT OF HIDDEN UNITS HERE
#There will be N+1 of them (Number of units + one bias unit, added later on)
#hA = np.zeros( (20+1) )
#hA = np.zeros( (50+1) )
hA = np.zeros( (100+1) )
#Uncomment when ready to use other hidden activations
#hA = np.zeros( (2+1) )

##OUTPUT ACTIVATION UNITS
#THERE SHOULD BE 10, ONE FOR EACH CLASSIFICATION
oA = np.zeros( (10) )

##ERROR CALCULATION VECTORS
#These are used to save the result of the error calculation
#they will then be used to change weights during back propagation
#outputError
oE = []

#hiddenError
hE = []

#WEIGHTS - remember that they only change during training
###INITIAL WEIGHTS - uncomment when ready
#Make sure to change the dimensions of the matrices when varying hidden unit amounts
itHW = np.random.uniform(low=-0.05, high=0.05, size=len(trainingData[0])*(len(hA)-1)).reshape(len(hA)-1,len(trainingData[0]))
htOW = np.random.uniform(low=-0.05, high=0.05, size=len(hA)*10).reshape(10,len(hA))

#These ones are N x M+1 where N are the number of hidden units
#and M+1 are the input units plus the bias input
# input_to_hidden_weights
'''
#len
itHW = [[-.4, .2, .1],
        [-.2, .4, -.1]]
'''
#These ones are K x N+1 where N are the number of hidden units
#and K are the number of output units (should be 10)
# hidden_to_output_weights
'''
htOW = [[.1, -.2, .1],
        [.4, -.1, .1]]
'''
#These weights will store the previous change in weights
###ONLY IF IMPLEMENTING MOMENTUM###
'''
itHWp = np.zeros( ( len(protoData) , len(hA) ) )
htOWp = np.zeros( ( 2 , len(hA) ) )
'''

#SIGMOID ACTIVATION FUNCTION
#Takes in a matrix of weights X inputs
#and gives us a 1xH or 1xO array of sigmoid outputs
#Where H are the number of hidden units and O the number of output units
def sigmoid ( z ):
    return np.divide( 1, np.add( 1,np.exp(-z) ) );

#ERROR(LOSS) FUNCTIONS
#Take in the 1x10 output vector and 1x10 target label
#and gives us a 1x10 array of error values
def errorOutput ( oA, t ):
    return np.multiply( oA, np.multiply( np.subtract(1,oA), np.subtract(t,oA) ) );

#Take in a 1xH hidden vector and 1x10 errorOutput vector
#and gives us a 1xH array of error values
def errorHidden ( hA, htOW, oE):
    
    return np.multiply(hA, np.multiply( np.subtract(1,hA), np.dot(np.transpose(htOW), oE) ) )[1:];

'''
print("\n---FORWARD PROPAGATION---\n")
print(protoData[0])
print(np.dot(itHW,protoData[0]))

print("\nHIDDEN UNITS UPDATED\n")
#hA = np.concatenate(([1],sigmoid(np.dot(itHW,protoData[0]))),axis=0)

#Use this for now, revert when question is answered
hA = np.around(np.concatenate(([1],sigmoid(np.dot(itHW,protoData[0]))),axis=0),2)
print( np.around( hA, 2))

print("\nOUTPUT UNITS UPDATED\n")
#oA = sigmoid(np.dot(htOW,hA))

#Use this for now, revert when question is answered
oA = np.around(sigmoid(np.dot(htOW,hA)),2)
print( np.around( oA, 2))

#SAVE THE OUTPUTS HERE


print("\nCALCULATE ERROR")
#Calculate error for each output and hidden unit
#oE = errorOutput( oA, protoLabels[0] )

#Use this for now, revert when question is answered
oE = np.around(errorOutput( oA, protoLabels[0] ), 3)
print( np.around( oE, 3))


#hE = errorHidden( hA, htOW, oE )

#Use this for now, revert when question is answered
hE = np.around(errorHidden( hA, htOW, oE ), 3)
print( np.around( hE, 3))

print("\n---BACKWARD PROPAGATION---\n")

print("\nUPDATE WEIGHTS")

#new hto weights = old hto weights + learning rate * o error * hidden units
#Do an outer multiplication between oE and hA so that we get back an HxO matrix
#Then the new weights would just be the old ones + the change

#Momentum version
#htOW = np.add( np.add(htOW , np.multiply( lR, np.outer( oE, hA ) ) ), htOWp )
#htOWp = np.multiply( lR, np.outer( oE, hA ) )

htOW = np.add(htOW , np.multiply( lR, np.outer( oE, hA ) ) )
print(htOW)


#updating the weights for inputs to hidden is similar

#Momentum version
#itHW = np.add( np.add(itHW , np.multiply( lR, np.outer( hE, protoData[0]) ) ), itHWp )
#itHWp = np.multiply( lR, np.outer( hE, protoData[0]) )

itHW = np.add(itHW , np.multiply( lR, np.outer( hE, protoData[0]) ) )
print(itHW)
'''


#There will be 50 Epochs
for e in range(50):

    #These arrays will keep track of each example's prediction
    #It will be appended with the label of the highest output value
    predictions = []
    predictionsTst = []

    #This array has data for the confusion matrix
    predictionsTstC = []

    #TRAINING
    #Where k is one of the dataset's examples
    for k in range(len(trainingData)):

        #print("\n---FORWARD PROPAGATION---\n")
        #We'll have the activations of the hidden layer now, remember to add the bias
        hA = np.concatenate(([1],sigmoid(np.dot(itHW,trainingData[k]))),axis=0)

        #print("\nOUTPUT UNITS UPDATED\n")
        oA = sigmoid(np.dot(htOW,hA))
  
        #SAVE THE OUTPUTS HERE
        
        predictions.append(np.argmax(oA))
        
        
        #print("\nCALCULATE ERROR")
        #Calculate error for each output and hidden unit
        oE = errorOutput( oA, targetTrainingData[k] )


        hE = errorHidden( hA, htOW, oE )

        #print("\n---BACKWARD PROPAGATION---\n")
        #print("\nUPDATE WEIGHTS")

        #new hto weights = old hto weights + learning rate * o error * hidden units
        #Do an outer multiplication between oE and hA so that we get back an HxO matrix
        #Then the new weights would just be the old ones + the change

        htOW = np.add(htOW , np.multiply( lR, np.outer( oE, hA ) ) )

        #updating the weights for inputs to hidden is similar
        itHW = np.add(itHW , np.multiply( lR, np.outer( hE, trainingData[k]) ) )

        ##We have gone through every example in the training set


    #TESTING
    for k in range(len(testingData)):

        #Now that the the machine is trained, pit it against test examples
        #print("\n---FORWARD PROPAGATION---\n")
        hA = np.concatenate(([1],sigmoid(np.dot(itHW,testingData[k]))),axis=0)

        #print("\nOUTPUT UNITS UPDATED\n")
        oA = sigmoid(np.dot(htOW,hA))
       
        #get the prediction of output for every example, 
        #SAVE THE OUTPUTS HERE
        predictionsTst.append(np.argmax(oA))

        #Get confusion matrix data only on the last epoch
        if e == 49:
            predictionsTstC.append(np.argmax(oA))

        #DO NOT CHANGE WEIGHTS
        

    ###DATA COLLECTION OVER

    
    

    ###ACCURACY CALCULATION
    #Take the accuracy and divide it over the correct guesses to find out how often the perceptrons got a right guess
    correctGuesses = 0
    correctGuessesTst = 0

    #Increase the amount of correct guesses for the training set
    for a in range(len(trainingData)):
        if np.array_equal(predictions[a],  targetTrainingLabels[a]):
            correctGuesses = correctGuesses + 1


    #Summarize confusion on the test set
    for a in range(len(testingData)):
    #Increase the confusion matrix cell for each respective guess vs target
        

        #Create the confusion matrix
        if e == 49:
            confusionMatrix[predictionsTstC[a]][targetTestingLabels[a]] = confusionMatrix[predictionsTstC[a]][targetTestingLabels[a]] + 1

        #Also increase the amount of correct guesses for the testing set
    
        if np.array_equal(predictionsTst[a],  targetTestingLabels[a]):
            correctGuessesTst = correctGuessesTst + 1
        
    accuracyFinal.append(correctGuesses/len(trainingData))
    accuracyFinalTst.append(correctGuessesTst/len(testingData))

    print("Epoch ", e, " --- %s seconds ---" % (time.time() - start_time))


#After all epochs have passed, display the confusion matrix and the accuracies of each epoch,
#This will summarize the test set

print("\nTraining Accuracy")
for acc in range(len(accuracyFinal)):
    print(accuracyFinal[acc])
    
print("\nTesting Accuracy")
for acc in range(len(accuracyFinalTst)):
    print(accuracyFinalTst[acc])
    
print("\nConfusion Matrix")
for cm in range(len(confusionMatrix)):
    print(confusionMatrix[cm])

print("Total time: --- %s seconds ---" % (time.time() - start_time))

