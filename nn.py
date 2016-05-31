# Neural Network implementation (via Welch Labs)
# 2 inputs, 1 hidden layer with 3 units, 1 output
# With sigmoid activation function

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

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
        
    def sigmoidPrime(self, z):
        #Derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        
        return dJdW1, dJdW2
        
#------Testing 1--------

##Define matrices for example input-output values
##X: (hours of sleep, hours studying)
##y: (score on test)
#X = np.array(([3,5], [5,1], [10,2]), dtype=float)
#y = np.array(([75], [82], [93]), dtype=float)
#
##scale the input and output (by dividing by max input or output value)
#X = X/np.amax(X, axis=0)
#y = y/100 #max test score is 100
#
##Make a Neural_Network object
#NN = Neural_Network()
#
##Get estimate of output, yHat, by calling forward function with input matrix, X
#yHat = NN.forward(X)

#------Testing 2--------

#nn = Neural_Network()
#testValues = np.arange(-5,5,0.01)
#plt.plot(testValues, nn.sigmoid(testValues), linewidth=2)
#plt.plot(testValues, nn.sigmoidPrime(testValues), linewidth=2)
#plt.grid(1)
#plt.legend(['sigmoid','sigmoidPrime'])
#plt.show()

#------Testing 3--------

#dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
#NN.W1 = NN.W1 - scalar.dJdW1
#NN.W2 = NN.W2 - scalar.dJdW2
#updatedCost = NN.costFunction(X,y)
 
 
class trainer(object):
    def __init__(self,N):
        #Make local reference to Neural Network:
        self.N = N
        
    #Wrapper function
    #To use BFGS, minimize function requires passing in an objective function
    #that accepts a vector of parameters, input and output data,
    #and returns both the cost and gradients
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
    
    def train(self, X, y):
        #Make initial variable for callback function:
        self.X = X
        self.y = y
        
        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams() #initial parameters
        options = {'maxiter':200, 'disp':True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, \
        jac=True, method='BFGS', args=(X,y), callback=self.callbackF)
        
         self.N.setParams(_res.x)
         self.optimizationResults = _res 
        