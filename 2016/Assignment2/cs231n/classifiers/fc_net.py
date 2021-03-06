import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, inputDim = 3 * 32 * 32, hiddenDim = 100, numClasses = 10,
			   weightScale = 1e-3, reg = 0.0):
	"""
	Initialize a new network.

	Inputs:
	- inputDim: An integer giving the size of the input
	- hiddenDim: An integer giving the size of the hidden layer
	- numClasses: An integer giving the number of classes to classify
	- dropout: Scalar between 0 and 1 giving dropout strength.
	- weightScale: Scalar giving the standard deviation for random
	  initialization of the weights.
	- reg: Scalar giving L2 regularization strength.
	"""
	self.params = {}
	self.reg = reg
	
	############################################################################
	# TODO: Initialize the weights and biases of the two-layer net. Weights    #
	# should be initialized from a Gaussian with standard deviation equal to   #
	# weightScale, and biases should be initialized to zero. All weights and  #
	# biases should be stored in the dictionary self.params, with first layer  #
	# weights and biases using the keys 'W1' and 'b1' and second layer weights #
	# and biases using the keys 'W2' and 'b2'.                                 #
	############################################################################
	
	## Initialising parameters.
	self.params['W1'] = np.random.normal(loc = 0, scale = weightScale, size = (inputDim, hiddenDim))
	self.params['b1'] = np.zeros(hiddenDim)

	self.params['W2'] = np.random.normal(loc = 0, scale = weightScale, size = (hiddenDim, numClasses))
	self.params['b2'] = np.zeros(numClasses)

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################


  def loss(self, X, y = None):
	"""
	Compute loss and gradient for a minibatch of data.

	Inputs:
	- X: Array of input data of shape (N, d_1, ..., d_k)
	- y: Array of labels, of shape (N,). y[i] gives the label for X[i].

	Returns:
	If y is None, then run a test-time forward pass of the model and return:
	- scores: Array of shape (N, C) giving classification scores, where
	  scores[i, c] is the classification score for X[i] and class c.

	If y is not None, then run a training-time forward and backward pass and
	return a tuple of:
	- loss: Scalar value giving the loss
	- grads: Dictionary with the same keys as self.params, mapping parameter
	  names to gradients of the loss with respect to those parameters.
	"""  
	scores = None
	############################################################################
	# TODO: Implement the forward pass for the two-layer net, computing the    #
	# class scores for X and storing them in the scores variable.              #
	############################################################################
	
	## Fully Connected + ReLU layer.
	hiddenRelu1, hiddenRelu1Cache = affineReluForward(X, self.params['W1'], self.params['b1'])

	## Fully Connected Layer.
	rawScores, rawScoresCache = affineForward(hiddenRelu1, self.params['W2'], self.params['b2'])

	## Storing the scores.
	scores = rawScores

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################

	# If y is None then we are in test mode so just return scores
	if y is None:
	  return scores
	
	loss, grads = 0, {}
	############################################################################
	# TODO: Implement the backward pass for the two-layer net. Store the loss  #
	# in the loss variable and gradients in the grads dictionary. Compute data #
	# loss using softmax, and make sure that grads[k] holds the gradients for  #
	# self.params[k]. Don't forget to add L2 regularization!                   #
	#                                                                          #
	# NOTE: To ensure that your implementation matches ours and you pass the   #
	# automated tests, make sure that your L2 regularization includes a factor #
	# of 0.5 to simplify the expression for the gradient.                      #
	############################################################################
	
	## Softmax Layer (Forward + Backward).
	loss, dscores = softmaxLoss(scores, y)

	## Add regularisation loss.
	loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])

	## Backprop through the second hidden layer.
	dHiddenRelu1, grads['W2'], grads['b2'] = affineBackward(dscores, rawScoresCache)

	## Backprop through the first hidden layer.
	dx, grads['W1'], grads['b1'] = affineReluBackward(dHiddenRelu1, hiddenRelu1Cache)

	## Adding regularisation to the weight gradients.
	grads['W1'] = grads['W1'] + self.reg * self.params['W1']
	grads['W2'] = grads['W2'] + self.reg * self.params['W2']

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################

	return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hiddenDims, inputDim = 3 * 32 * 32, numClasses = 10,
			   dropout = 0, useBatchnorm = False, reg = 0.0,
			   weightScale = 1e-2, dtype = np.float32, seed = None):
	"""
	Initialize a new FullyConnectedNet.
	
	Inputs:
	- hiddenDims: A list of integers giving the size of each hidden layer.
	- inputDim: An integer giving the size of the input.
	- numClasses: An integer giving the number of classes to classify.
	- dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
	  the network should not use dropout at all.
	- useBatchnorm: Whether or not the network should use batch normalization.
	- reg: Scalar giving L2 regularization strength.
	- weightScale: Scalar giving the standard deviation for random
	  initialization of the weights.
	- dtype: A numpy datatype object; all computations will be performed using
	  this datatype. float32 is faster but less accurate, so you should use
	  float64 for numeric gradient checking.
	- seed: If not None, then pass this random seed to the dropout layers. This
	  will make the dropout layers deteriminstic so we can gradient check the
	  model.
	"""
	self.useBatchnorm = useBatchnorm
	self.use_dropout = dropout > 0
	self.reg = reg
	self.num_layers = 1 + len(hiddenDims)
	self.dtype = dtype
	self.params = {}
	self.hiddenDims = hiddenDims

	############################################################################
	# TODO: Initialize the parameters of the network, storing all values in    #
	# the self.params dictionary. Store weights and biases for the first layer #
	# in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
	# initialized from a normal distribution with standard deviation equal to  #
	# weightScale and biases should be initialized to zero.                   #
	#                                                                          #
	# When using batch normalization, store scale and shift parameters for the #
	# first layer in gamma1 and beta1; for the second layer use gamma2 and     #
	# beta2, etc. Scale parameters should be initialized to one and shift      #
	# parameters should be initialized to zero.                                #
	############################################################################
	
	## Initialising parameters.
	
	## Creating a list to contain the sizes of all the layers.
	layerSizes = [inputDim] + hiddenDims + [numClasses]

	## Initialising hidden layer parameters.
	for i in range(len(layerSizes) - 1):

		self.params['W' + str(i + 1)] = np.random.normal(loc = 0, scale = weightScale, size = (layerSizes[i], layerSizes[i+1]))
		self.params['b' + str(i + 1)] = np.zeros(layerSizes[i+1])

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################

	# When using dropout we need to pass a dropout_param dictionary to each
	# dropout layer so that the layer knows the dropout probability and the mode
	# (train / test). You can pass the same dropout_param to each dropout layer.
	self.dropout_param = {}
	if self.use_dropout:
	  self.dropout_param = {'mode': 'train', 'p': dropout}
	  if seed is not None:
		self.dropout_param['seed'] = seed
	
	# With batch normalization we need to keep track of running means and
	# variances, so we need to pass a special bn_param object to each batch
	# normalization layer. You should pass self.bn_params[0] to the forward pass
	# of the first batch normalization layer, self.bn_params[1] to the forward
	# pass of the second batch normalization layer, etc.
	self.bn_params = []
	if self.useBatchnorm:
		self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

		## Initialize gamma and beta parameters to 1 and 0 respectively for each batch normalization layer.
		for i in range(self.num_layers - 1):
			self.params['gamma' + str(i + 1)] = np.ones(layerSizes[i+1])
			self.params['beta' + str(i + 1)] = np.zeros(layerSizes[i+1])
	
	# Cast all parameters to the correct datatype
	for k, v in self.params.iteritems():
	  self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
	"""
	Compute loss and gradient for the fully-connected net.

	Input / output: Same as TwoLayerNet above.
	"""
	X = X.astype(self.dtype)
	mode = 'test' if y is None else 'train'

	# Set train/test mode for batchnorm params and dropout param since they
	# behave differently during training and testing.
	if self.dropout_param is not None:
	  self.dropout_param['mode'] = mode   
	if self.useBatchnorm:
	  for bn_param in self.bn_params:
		bn_param[mode] = mode

	scores = None
	############################################################################
	# TODO: Implement the forward pass for the fully-connected net, computing  #
	# the class scores for X and storing them in the scores variable.          #
	#                                                                          #
	# When using dropout, you'll need to pass self.dropout_param to each       #
	# dropout forward pass.                                                    #
	#                                                                          #
	# When using batch normalization, you'll need to pass self.bn_params[0] to #
	# the forward pass for the first batch normalization layer, pass           #
	# self.bn_params[1] to the forward pass for the second batch normalization #
	# layer, etc.                                                              #
	############################################################################
	
	## Creating a dictionary to store the layer outputs and cache.
	outputs = {}
	cache = {}

	## Computing outputs and cache of hidden layers.
	for i in range(0, len(self.hiddenDims)):
		
		## If Batch Normalization is to be activated.
		if self.useBatchnorm:
			outputs['hiddenLayer' + str(i+1)], cache['hiddenLayer' + str(i+1)] = affineBatchNormReLuForward(X, self.params['W' + str(i+1)], self.params['b'+ str(i+1)], self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)], self.bn_params[i])
		else:
			outputs['hiddenLayer' + str(i+1)], cache['hiddenLayer' + str(i+1)] = affineReluForward(X, self.params['W' + str(i+1)], self.params['b'+ str(i+1)])

		if self.use_dropout:
			outputs['hiddenLayerDrop' + str(i+1)], cache['hiddenLayerDrop' + str(i+1)] = dropoutForward((outputs['hiddenLayer' + str(i+1)]), self.dropout_param)
			X = outputs['hiddenLayerDrop' + str(i+1)]    
		else:
			X = outputs['hiddenLayer' + str(i+1)]
		
		
	## Computing outputs and cache of the last fully connected layer.
	outputs['lastFC'], cache['lastFC'] = affineForward(X, self.params['W' + str(i+2)], self.params['b'+ str(i+2)])

	## Updating scores.
	scores = outputs['lastFC']
		
	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################

	# If test mode return early
	if mode == 'test':
	  return scores

	loss, grads = 0.0, {}
	############################################################################
	# TODO: Implement the backward pass for the fully-connected net. Store the #
	# loss in the loss variable and gradients in the grads dictionary. Compute #
	# data loss using softmax, and make sure that grads[k] holds the gradients #
	# for self.params[k]. Don't forget to add L2 regularization!               #
	#                                                                          #
	# When using batch normalization, you don't need to regularize the scale   #
	# and shift parameters.                                                    #
	#                                                                          #
	# NOTE: To ensure that your implementation matches ours and you pass the   #
	# automated tests, make sure that your L2 regularization includes a factor #
	# of 0.5 to simplify the expression for the gradient.                      #
	############################################################################

	## Computing the loss and the gradient for the softmax layer.
	loss, dscores = softmaxLoss(scores, y)

	## Adding regularisation to the loss.
	for j in range(0, len(self.hiddenDims) + 1):		
		loss += 0.5 * self.reg * np.sum(self.params['W' + str(j+1)] * self.params['W' + str(j+1)])

	## Performing backprop on the last fully connected layer.
	dLastFC, grads['W' + str(i+2)], grads['b' + str(i+2)] = affineBackward(dscores, cache['lastFC'])
	grads['W' + str(i+2)] += self.reg * self.params['W' + str(i+2)]

	## Performing backprop on the hidden layers.
	for i in range(len(self.hiddenDims), 0, -1):
		
		if self.use_dropout:
			dHiddenDropout = dropoutBackward(dLastFC, cache['hiddenLayerDrop' + str(i)])
			dHiddenRelu, grads['W' + str(i)], grads['b' + str(i)] = affineReluBackward(dHiddenDropout, cache['hiddenLayer' + str(i)])

		if self.useBatchnorm:
			dHiddenRelu, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = affineBatchNormReLuBackward(dLastFC, cache['hiddenLayer' + str(i)])

		else:
			dHiddenRelu, grads['W' + str(i)], grads['b' + str(i)] = affineReluBackward(dLastFC, cache['hiddenLayer' + str(i)])
			
		grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
		dLastFC = dHiddenRelu

	############################################################################
	#                             END OF YOUR CODE                             #
	############################################################################

	return loss, grads


## Defining a utility function to encapsulate the Affine Transformation followed by 
## Batch Normalization follwed by a Relu non-linearity in a single function.
def affineBatchNormReLuForward(x, w, b, gamma, beta, bn_param):

	## Performing the Affine Transformation.
	outPut, fcCache = affineForward(x, w, b)

	## Applying the forward pass of Batch Normalization.
	outPut, batchNormCache = batchnormForward(outPut, gamma, beta, bn_param)

	## Applying the reLu non-linearity.
	outPut, reLuCache = reluForward(outPut)

	cache = (fcCache, batchNormCache, reLuCache)

	return outPut, cache

## Defining a utility function to compute the backward pass for the transformations defined
## in the above function.
def affineBatchNormReLuBackward(dOut, cache):

	## Unpacking cache.
	fcCache, batchNormCache, reLuCache = cache

	## Backprop through the reLu layer.
	dRelu = relu_backward(dOut, reLuCache)

	## Backprop through the Batch Normalization layer.
	dBatchNorm, dGamma, dBeta = batchnorm_backward(dRelu, batchNormCache)

	## Backprop through the Affine Transformation layer.
	dAffineBackward, dW, dB = affine_backward(dBatchNorm, fcCache)

	return dAffineBackward, dW, dB, dGamma, dBeta

