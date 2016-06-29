# 6-24-2016: Rewrite draft of CNN with numpy for incremental testing.
# -------------------------------------------------------------------
# Notes:
# - Array indexing: correct convention is (Depth, Height, Width)
# - Padding and stride not yet taken into account
# - Using Python 2.7
# 
# To do:
# - Apply element-wise activation function after convolving. ReLU/tanh
# - ReLu/tanh/sigmoid activation functions after fully-connected layers
# - Softmax vs. logistic regression
# - Backpropagation in FC architecture?
# -------------------------------------------------------------------

import numpy as np

def generateRandomKernels(inputDepth, numKernels, kernelSize):
	return [np.random.rand(inputDepth, kernelSize, kernelSize) for i in range(numKernels)]



def convLayer(inputVolume, kernels):
	print 'Putting input through convolutional layer.'

	kernelSize = np.shape(kernels[0])[2]

	inputHeight = np.shape(inputVolume)[1]
	inputWidth = np.shape(inputVolume)[2]

	outputHeight = inputHeight - kernelSize + 1
	outputWidth = inputWidth - kernelSize + 1
	
	if kernelSize > outputHeight or kernelSize > outputWidth:
		raise ValueError('Kernel size is larger than output volume height or width.')

	print 'kernel size:', kernelSize
	print 'output volume height:', outputHeight
	print 'output volume width:', outputWidth

	outputVolume = np.zeros((len(kernels), outputHeight, outputWidth))
	
	print 'expected output volume shape:', np.shape(outputVolume)

	for kernelIndex in range(len(kernels)):
		for i in range(outputHeight):
			for j in range(outputWidth):
				featureMap = np.sum(inputVolume[:, i:i+kernelSize, j:j+kernelSize] * kernels[kernelIndex])
				outputVolume[kernelIndex, i, j] = featureMap


	return np.tanh(outputVolume)




def poolLayer(inputVolume, poolFactor):
	print 'Putting input through pooling layer.'

	inputDepth = np.shape(inputVolume)[0]
	inputHeight = np.shape(inputVolume)[1]
	inputWidth = np.shape(inputVolume)[2]

	if not ((inputHeight % poolFactor == 0) and (inputWidth % poolFactor == 0)):
		raise ValueError('Pool factor does not neatly divide the input volume height and weight.')

	outputHeight = inputHeight / poolFactor
	outputWidth = inputWidth / poolFactor

	maxIndexMatrix = np.zeros((inputDepth, outputHeight, outputWidth))
	outputVolume = np.zeros((inputDepth, outputHeight, outputWidth))

	print 'expected output volume shape', np.shape(outputVolume)

	for layerIndex in range(inputDepth):
		for i in range(outputHeight):
			for j in range(outputWidth):
				poolWindow = inputVolume[
								layerIndex,
								i*poolFactor : i*poolFactor+poolFactor,
								j*poolFactor : j*poolFactor+poolFactor]
				maxVal = np.amax(poolWindow)
				outputVolume[layerIndex, i, j] = maxVal

	return outputVolume


def generateRandomWeights(numInputNeurons, numOutputNeurons):
	return np.random.rand(numInputNeurons, numOutputNeurons)


def fcLayer(X, W, activation):
	#Note: make sure to concatenate another neuron, for bias

	print 'Putting input through FC layer.'
	
	Z = np.dot(X, W)

	if activation=='tanh':
		return np.tanh(Z)
	elif activation=='relu':
		return np.maximum(0, Z)
	else:
		raise ValueError('Invalid activation function option.')


if __name__ == "__main__":
	print "Testing."

	#Layer 0: Random input volume of dimension 3x64x64
	inputVolume = np.random.rand(3, 64, 64)

	#Layer 1: Convolutional layer with 10 filters of dimension 3x3
	kernels1 = generateRandomKernels(inputDepth=np.shape(inputVolume)[0], numKernels=10, kernelSize=3)
	outputVolume_conv1 = convLayer(inputVolume=inputVolume, kernels=kernels1)
	print 'resulting output volume shape:', np.shape(outputVolume_conv1)

	#Layer 2: Pooling layer with pool factor of 2
	outputVolume_pool1 = poolLayer(inputVolume=outputVolume_conv1, poolFactor=2)
	print 'resulting output volume shape:', np.shape(outputVolume_pool1)

	#Layer 3: Convolutional layer with 10 filters of dimension 4x4
	kernels2 = generateRandomKernels(inputDepth=np.shape(outputVolume_pool1)[0], numKernels=10, kernelSize=4)
	outputVolume_conv2 = convLayer(inputVolume=outputVolume_pool1, kernels=kernels2)
	print 'resulting output volume shape:', np.shape(outputVolume_conv2)

	#Layer 4: Pooling layer with pool factor of 2
	outputVolume_pool2 = poolLayer(inputVolume=outputVolume_conv2, poolFactor=2)
	print 'resulting output volume shape:', np.shape(outputVolume_pool2)

	#Fully-connected layers: FC input layer, Hidden layer with 12 neurons, FC output layer with 10 class scores

	#Layer 5: FC input layer
	fcInputDim = np.shape(outputVolume_pool2)[0] * np.shape(outputVolume_pool2)[1] * np.shape(outputVolume_pool2)[2] 
	inputVector = outputVolume_pool2.ravel().reshape((1, fcInputDim))

	weightMatrix1 = generateRandomWeights(numInputNeurons=fcInputDim, numOutputNeurons=12)
	outputMatrix_fc1 = fcLayer(X=inputVector, W=weightMatrix1, activation='tanh')

	#Layer 6: Fully connected hidden layer
	weightMatrix2 = generateRandomWeights(numInputNeurons=len(outputMatrix_fc1[0]), numOutputNeurons=10)
	outputMatrix_fc2 = fcLayer(X=outputMatrix_fc1, W=weightMatrix2, activation='tanh')

	#Layer 7: Fully connected output layer (class scores output)
	print outputMatrix_fc2