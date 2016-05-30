# Neural Network implementation (via Welch Labs)
# 2 inputs, 1 hidden layer with 3 units, 1 output
# With sigmoid activation function

import numpy as np

class Neural_Network(object):
    def __init__(self):
        #Define hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (Parameters)
        #first set weights as random numbers. (to be adjusted during learning)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
        
#------Testing--------

#Define matrices for example input-output values
#X: (hours of sleep, hours studying)
#y: (score on test)
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

#scale the input and output (by dividing by max input or output value)
X = X/np.amax(X, axis=0)
y = y/100 #max test score is 100

#Make a Neural_Network object
NN = Neural_Network()

#Get estimate of output, yHat, by calling forward function with input matrix, X
yHat = NN.forward(X)

