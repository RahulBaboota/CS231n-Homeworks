import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, inputDim = (3, 32, 32), numFilters = 32, filterSize = 7,
			   hiddenDim = 100, numClasses = 10, weightScale = 1e-3, reg = 0.0,
			   dtype = np.float32):
	"""
	Initialize a new network.
	
	Inputs:
	- input_dim: Tuple (C, H, W) giving size of input data
	- num_filters: Number of filters to use in the convolutional layer
	- filter_size: Size of filters to use in the convolutional layer
	- hidden_dim: Number of units to use in the fully-connected hidden layer
	- num_classes: Number of scores to produce from the final affine layer.
	- weight_scale: Scalar giving standard deviation for random initialization
	  of weights.
	- reg: Scalar giving L2 regularization strength
	- dtype: numpy datatype to use for computation.
	"""
	self.params = {}
	self.reg = reg
	self.dtype = dtype
	
	############################################################################
	# TODO: Initialize weights and biases for the three-layer convolutional    #
	# network. Weights should be initialized from a Gaussian with standard     #
	# deviation equal to weight_scale; biases should be initialized to zero.   #
	# All weights and biases should be stored in the dictionary self.params.   #
	# Store weights and biases for the convolutional layer using the keys 'W1' #
	# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
	# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
	# of the output affine layer.                                              #
	############################################################################
	
	## Initialising parameters.
	self.params['W1'] = np.random.normal(loc = 0, scale = weightScale, size = (numFilters, inputDim[0], filterSize, filterSize))
	self.params['b1'] = np.zeros(numFilters)

	## Assuming the output of the convolutional layer is the same as the input and then reduced by a factor of 4 after applying 2*2 max pooling.
	self.params['W2'] = np.random.normal(loc = 0, scale = weightScale, size = ((numFilters * inputDim[1] * inputDim[2]) / 4, hiddenDim))
	self.params['b2'] = np.zeros(hiddenDim)

	self.params['W3'] = np.random.normal(loc = 0, scale = weightScale, size = (hiddenDim, numClasses))
	self.params['b3'] = np.zeros(numClasses)

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################

	for k, v in self.params.iteritems():
	  self.params[k] = v.astype(dtype)
	 
 
  def loss(self, X, y = None):
	"""
	Evaluate loss and gradient for the three-layer convolutional network.
	
	Input / output: Same API as TwoLayerNet in fc_net.py.
	"""
	W1, b1 = self.params['W1'], self.params['b1']
	W2, b2 = self.params['W2'], self.params['b2']
	W3, b3 = self.params['W3'], self.params['b3']
	
	# pass conv_param to the forward pass for the convolutional layer
	numFilters = W1.shape[0]
	filterSize = W1.shape[2]
	convParam = {'stride': 1, 'pad': (filterSize - 1) / 2}

	# pass pool_param to the forward pass for the max-pooling layer
	poolParam = {'poolHeight': 2, 'poolWidth': 2, 'stride': 2}

	scores = None
	############################################################################
	# TODO: Implement the forward pass for the three-layer convolutional net,  #
	# computing the class scores for X and storing them in the scores          #
	# variable.                                                                #
	############################################################################
	
	## Applying the convolutional layer, followed by a reLu non-linearity which
	## is then followed by a max-pooling layer.
	maxPoolOut, maxPoolCache = conv_relu_pool_forward(X, W1, b1, convParam, poolParam)

	## Reshaping the above output so that affine transformation (fully connected 
	## layers) can be used.
	maxPoolOut = maxPoolOut.reshape(maxPoolOut.shape[0], maxPoolOut.shape[1] * maxPoolOut.shape[2] * maxPoolOut.shape[3])

	## Applying the affine transformation.
	reLuOut, reLuCache = affine_relu_forward(maxPoolOut, W2, b2)    

	## Applying the final affine transformation.
	fcOut, fcCache = affine_forward(reLuOut, W3, b3)

	scores = fcOut

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################
	
	if y is None:
	  return scores
	
	loss, grads = 0, {}
	############################################################################
	# TODO: Implement the backward pass for the three-layer convolutional net, #
	# storing the loss and gradients in the loss and grads variables. Compute  #
	# data loss using softmax, and make sure that grads[k] holds the gradients #
	# for self.params[k]. Don't forget to add L2 regularization!               #
	############################################################################

	## Softmax Layer (Forward + Backward).
	loss, dScores = softmax_loss(scores, y)

	## Adding regularisation to the loss.
	for j in range(0, 3):
		loss += 0.5 * self.reg * np.sum(self.params['W' + str(j + 1)] * self.params['W' + str(j + 1)])

	## Backproping through the last fully-connected layer.
	dReluOut, grads['W3'], grads['b3'] = affine_backward(dScores, fcCache)

	## Backproping through the hidden layer.
	dMaxPoolOut, grads['W2'], grads['b2'] = affine_relu_backward(dReluOut, reLuCache)

	## Reshaping the gradient matrix.
	dMaxPoolOut = dMaxPoolOut.reshape(dMaxPoolOut.shape[0], numFilters, X.shape[2] / 2, X.shape[3] / 2)

	## Backproping through the convolutional layer.
	dX, grads['W1'], grads['b1'] = conv_relu_pool_backward(dMaxPoolOut, maxPoolCache)

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################
	
	return loss, grads
  
  
pass
