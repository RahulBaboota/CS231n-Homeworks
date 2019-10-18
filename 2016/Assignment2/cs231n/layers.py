import numpy as np


def affineForward(x, W, b):
	"""
	Computes the forward pass for an affine (fully-connected) layer.

	The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
	examples, where each example x[i] has shape (d_1, ..., d_k). We will
	reshape each input into a vector of dimension D = d_1 * ... * d_k, and
	then transform it to an output vector of dimension M.

	Inputs:
	- x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
	- w: A numpy array of weights, of shape (D, M)
	- b: A numpy array of biases, of shape (M,)
	
	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w, b)
	"""
	out = None

	#############################################################################
	# TODO: Implement the affine forward pass. Store the result in out. You     #
	# will need to reshape the input into rows.                                 #
	#############################################################################

	## Defining the number of inputs.
	numInputs = x.shape[0]

	## Defining D = d_1 * d_2 * ..... * d_k
	D = np.prod(np.array(x.shape[1 : ]))

	## Making a copy of x for performing vectorized operations.
	xCopy = np.copy(x)

	## Reshaping the input vector to dimensions (N * D).
	xCopy = xCopy.reshape(numInputs, D)

	## Performing forward pass of fully connected layer.
	out = xCopy.dot(W) + b
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, W, b)
	return out, cache


def affineBackward(dout, cache):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
		- x: Input data, of shape (N, d_1, ... d_k)
		- w: Weights, of shape (D, M)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""

	x, W, b = cache
	dx, dW, db = None, None, None

	#############################################################################
	# TODO: Implement the affine backward pass.                                 #
	#############################################################################

	## Defining the number of inputs.
	numInputs = x.shape[0]
			
	## Defining D = d_1 * d_2 * ..... * d_k
	D = np.prod(np.array(x.shape[1 : ]))

	## Making a copy of x for performing vectorized operations.
	xCopy = np.copy(x)

	## Computing the derivative wrt Biases.
	db = np.sum(dout, axis = 0)

	## Reshaping the input vector to dimensions (N * D).
	xCopy = xCopy.reshape(numInputs, D)

	## Computing the derivative wrt Weights.
	dW = xCopy.T.dot(dout)

	## Computing the derivative wrt Input.
	dx = dout.dot(W.T)

	## Reshaping dx.
	dx = dx.reshape(x.shape)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx, dW, db


def reluForward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = None
	#############################################################################
	# TODO: Implement the ReLU forward pass.                                    #
	#############################################################################

	## Apply the ReLU function -> f : max(0,x).
	out = np.clip(x, 0, None)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = x
	return out, cache


def reluBackward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	dx, x = None, cache
	#############################################################################
	# TODO: Implement the ReLU backward pass.                                   #
	#############################################################################
	
	## BackProp through ReLU gate.
	dout[x <= 0] = 0

	dx = dout

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx


def batchnormForward(x, gamma, beta, bnParam):
	"""
	Forward pass for batch normalization.
	
	During training the sample mean and (uncorrected) sample variance are
	computed from minibatch statistics and used to normalize the incoming data.
	During training we also keep an exponentially decaying running mean of the mean
	and variance of each feature, and these averages are used to normalize data
	at test-time.

	At each timestep we update the running averages for mean and variance using
	an exponential decay based on the momentum parameter:

	runningMean = momentum * runningMean + (1 - momentum) * sampleMean
	runningVar = momentum * runningVar + (1 - momentum) * sampleVar

	Note that the batch normalization paper suggests a different test-time
	behavior: they compute sample mean and variance for each feature using a
	large number of training images rather than using a running average. For
	this implementation we have chosen to use running averages instead since
	they do not require an additional estimation step; the torch7 implementation
	of batch normalization also uses running averages.

	Input:
	- x: Data of shape (N, D)
	- gamma: Scale parameter of shape (D,)
	- beta: Shift paremeter of shape (D,)
	- bnParam: Dictionary with the following keys:
		- mode: 'train' or 'test'; required
		- eps: Constant for numeric stability
		- momentum: Constant for running mean / variance.
		- runningMean: Array of shape (D,) giving running mean of features
		- runningVar Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: of shape (N, D)
	- cache: A tuple of values needed in the backward pass
	"""
	mode = bnParam['mode']
	eps = bnParam.get('eps', 1e-5)
	momentum = bnParam.get('momentum', 0.9)

	N, D = x.shape
	runningMean = bnParam.get('runningMean', np.zeros(D, dtype = x.dtype))
	runningVar = bnParam.get('runningVar', np.zeros(D, dtype = x.dtype))

	out, cache = None, None

	if mode == 'train':
		#############################################################################
		# TODO: Implement the training-time forward pass for batch normalization.   #
		# Use minibatch statistics to compute the mean and variance, use these      #
		# statistics to normalize the incoming data, and scale and shift the        #
		# normalized data using gamma and beta.                                     #
		#                                                                           #
		# You should store the output in the variable out. Any intermediates that   #
		# you need for the backward pass should be stored in the cache variable.    #
		#                                                                           #
		# You should also use your computed sample mean and variance together with  #
		# the momentum variable to update the running mean and running variance,    #
		# storing your result in the runningMean and runningVar variables.        #
		#############################################################################
		
		## Computing the forward pass.
		## We compute the forward pass as a computational graph so as to easily 
		## backpropagate into the network. 

		## Computing the forward pass.
		## We compute the forward pass as a computational graph so as to easily 
		## backpropagate into the network. 

		## Computing the mean of the sample.
		sampleMean = (1.0/N) * (np.sum(x, axis = 0))

		## Computing the numerator expression (X - E[X]).
		numExpression = x - sampleMean

		## Computing the denominator expression (Standard Deviation of X).
		interMediate = numExpression ** 2
		sampleVariance = (1.0/N) * (np.sum(interMediate, axis = 0))
		stableSD = np.sqrt(sampleVariance + eps)

		## Inverting the standard deviation.
		invertedSD = (1.0/stableSD)

		## Computing the normalised gaussian input.
		xHat = numExpression * invertedSD

		## Scaling the normalised gaussian input by gamma.
		xHatScaled = gamma * xHat

		## Shifting the normalised gaussian input by beta.
		xHatShifted = xHatScaled + beta
		out = xHatShifted

		## Updating the running mean and running variance.
		runningMean = momentum * runningMean + (1 - momentum) * sampleMean
		runningVar = momentum * runningVar + (1 - momentum) * sampleVariance

		## Storing important information in cache to be used for backward pass.
		cache = (out, xHatShifted, xHatScaled, xHat, invertedSD, stableSD, sampleVariance, interMediate, numExpression, sampleMean, gamma, beta, eps)

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
	elif mode == 'test':
		#############################################################################
		# TODO: Implement the test-time forward pass for batch normalization. Use   #
		# the running mean and variance to normalize the incoming data, then scale  #
		# and shift the normalized data using gamma and beta. Store the result in   #
		# the out variable.                                                         #
		#############################################################################
		
		## Normalizing the input.
		out = ((x - runningMean)/np.sqrt(runningVar))

		## Scaling and Shifting the normalized data.
		out = (gamma * out + beta)

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	# Store the updated running means back into bnParam
	bnParam['runningMean'] = runningMean
	bnParam['runningVar'] = runningVar

	return out, cache


def batchnormBackward(dOut, cache):
	"""
	Backward pass for batch normalization.
	
	For this implementation, you should write out a computation graph for
	batch normalization on paper and propagate gradients backward through
	intermediate nodes.
	
	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from batchnorm_forward.
	
	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	N,D = dOut.shape
	dx, dgamma, dbeta = None, None, None
	out, xHatShifted, xHatScaled, xHat, invertedSD, stableSD, sampleVariance, interMediate, numExpression, sampleMean, gamma, beta, eps = cache

	#############################################################################
	# TODO: Implement the backward pass for batch normalization. Store the      #
	# results in the dx, dgamma, and dbeta variables.                           #
	#############################################################################

	## Implementing the backward pass.

	## Computing the gradient with respect to the beta parameter.
	dbeta = np.sum(dOut, axis = 0)

	## Computing the gradient with respect to the gamma parameter.
	dgamma = np.sum(xHat * dOut, axis = 0)

	## Computing the gradient with respect to xHat.
	dXhat = (gamma * dOut)

	## Computing the gradient with respect to inverted standard deviation.
	dInvertedSD = np.sum( numExpression * dXhat, axis = 0)

	## Computing the gradient with respect to the numerator expression (P1).
	dNumExpressionP1 = (invertedSD * dXhat)

	## Computing the gradient with respect to the standard deviation.
	dStableSD = ((-1.0) * (invertedSD**2) * dInvertedSD)

	## Computing the gradient with respect to the sample variance.
	dSampleVariance = ((0.5) * (1.0 / np.sqrt(sampleVariance + eps)) * dStableSD)

	## Computing the gradient with respect to the interMediate.
	dInterMediate = ((1.0 / N) * np.ones((N, D)) * dSampleVariance)

	## Computing the gradient with respect to the numerator expression (P2).
	dNumExpressionP2 = ((2.0) * numExpression * dInterMediate)

	## Combining the gradients to obtain the full gradient with respect to the numerator expression.
	dNumExpression = dNumExpressionP1  + dNumExpressionP2

	## Computing the gradient with respect to the sample mean.
	dSampleMean = (-1) * np.sum(dNumExpression, axis = 0)

	## Computing the gradient with respect to the input.
	dXP1 = ((1.0 / N) * np.ones((N, D)) * dSampleMean)
	dXP2 = dNumExpression
	dx = dXP1 + dXP2

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return dx, dgamma, dbeta


def batchnormBackwardAlt(dOut, cache):
	"""
	Alternative backward pass for batch normalization.
	
	For this implementation you should work out the derivatives for the batch
	normalizaton backward pass on paper and simplify as much as possible. You
	should be able to derive a simple expression for the backward pass.
	
	Note: This implementation should expect to receive the same cache variable
	as batchnorm_backward, but might not use all of the values in the cache.
	
	Inputs / outputs: Same as batchnorm_backward
	"""
	N,D = dOut.shape
	dx, dgamma, dbeta = None, None, None
	out, xHatShifted, xHatScaled, xHat, invertedSD, stableSD, sampleVariance, interMediate, numExpression, sampleMean, gamma, beta, eps = cache
	
	#############################################################################
	# TODO: Implement the backward pass for batch normalization. Store the      #
	# results in the dx, dgamma, and dbeta variables.                           #
	#                                                                           #
	# After computing the gradient with respect to the centered inputs, you     #
	# should be able to compute gradients with respect to the inputs in a       #
	# single statement; our implementation fits on a single 80-character line.  #
	#############################################################################
	

	## Computing the gradient with respect to the beta parameter.
	dbeta = np.sum(dOut, axis = 0)

	## Computing the gradient with respect to the gamma parameter.
	dgamma = np.sum(xHat * dOut, axis = 0)

	## Computing the gradient with respect to xHat.
	dXhat = (gamma * dOut)

	## Computing the gradient with respect to the input.
	dx = (1. / N) * invertedSD * ( (N * dXhat) - (np.sum(dXhat, axis = 0)) - (xHat * np.sum(xHat * dXhat, axis = 0)))

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	
	return dx, dgamma, dbeta


def dropoutForward(x, dropoutParam):
	"""
	Performs the forward pass for (inverted) dropout.

	Inputs:
	- x: Input data, of any shape
	- dropoutParam: A dictionary with the following keys:
		- p: Dropout parameter. We drop each neuron output with probability p.
		- mode: 'test' or 'train'. If the mode is train, then perform dropout;
			if the mode is test, then just return the input.
		- seed: Seed for the random number generator. Passing seed makes this
			function deterministic, which is needed for gradient checking but not in
			real networks.

	Outputs:
	- out: Array of the same shape as x.
	- cache: A tuple (dropoutParam, mask). In training mode, mask is the dropout
		mask that was used to multiply the input; in test mode, mask is None.
	"""
	p, mode = dropoutParam['p'], dropoutParam['mode']
	if 'seed' in dropoutParam:
		np.random.seed(dropoutParam['seed'])

	mask = None
	out = None

	if mode == 'train':
		###########################################################################
		# TODO: Implement the training phase forward pass for inverted dropout.   #
		# Store the dropout mask in the mask variable.                            #
		###########################################################################
		
		## Creating the mask.
		mask = (np.random.rand(*x.shape) < p) / p

		## Storing the dropped layer in the output variable.
		out = x * mask

		###########################################################################
		#                            END OF YOUR CODE                             #
		###########################################################################
	elif mode == 'test':
		###########################################################################
		# TODO: Implement the test phase forward pass for inverted dropout.       #
		###########################################################################
		 
		mask = None 
		out = x

		###########################################################################
		#                            END OF YOUR CODE                             #
		###########################################################################

	cache = (dropoutParam, mask)
	out = out.astype(x.dtype, copy = False)

	return out, cache


def dropoutBackward(dout, cache):
	"""
	Perform the backward pass for (inverted) dropout.

	Inputs:
	- dout: Upstream derivatives, of any shape
	- cache: (dropoutParam, mask) from dropout_forward.
	"""
	dropoutParam, mask = cache
	mode = dropoutParam['mode']
	
	dx = None
	if mode == 'train':
		###########################################################################
		# TODO: Implement the training phase backward pass for inverted dropout.  #
		###########################################################################
		
		dx = mask * dout

		###########################################################################
		#                            END OF YOUR CODE                             #
		###########################################################################
	elif mode == 'test':
		dx = dout
	return dx


def convForwardNaive(x, w, b, convParam):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and width
	W. We convolve each input with F different filters, where each filter spans
	all C channels and has height HH and width HH.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- convParam: A dictionary with the following keys:
		- 'stride': The number of pixels between adjacent receptive fields in the
			horizontal and vertical directions.
		- 'pad': The number of pixels that will be used to zero-pad the input.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
		H' = 1 + (H + 2 * pad - HH) / stride
		W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None

	#############################################################################
	# TODO: Implement the convolutional forward pass.                           #
	# Hint: you can use the function np.pad for padding.                        #
	#############################################################################

	## Unpacking the convolutional parameters.
	stride, padWidth = convParam['stride'], convParam['pad']

	## Padding the input with zeroes.
	xPadded = np.pad(x, pad_width = ((0, 0), (0, 0), (padWidth, padWidth), (padWidth, padWidth)), mode = 'constant', constant_values = 0)
	
	## Defining the input size, depth, height and width.
	inputSize = x.shape[0]
	inputDepth = x.shape[1]
	inputHeight = x.shape[2]
	inputWidth = x.shape[3]

	## Defining the filter height, width, depth and number of filters.
	numFilters = w.shape[0]
	filerDepth = w.shape[1]
	filterHeight = w.shape[2]
	filterWidth = w.shape[3]

	## Defining the output size, depth, height and width.
	outputSize = x.shape[0]
	outputDepth = numFilters
	outputHeight = ((inputHeight - filterHeight + 2 * padWidth) / stride + 1)
	outputWidth = ((inputWidth - filterWidth + 2 * padWidth) / stride + 1)

	## Initializing the output activation map.
	outputActivationMap = np.empty([outputSize, outputDepth, outputHeight, outputWidth])

	## Obtaining the necessary input slices over which
	## the weight matrices will convolve.
	for n in range(0, inputSize):
		for k in range(0, numFilters):
			for i in range(0, outputHeight):
				for j in range(0, outputWidth):

					## Obtaining the input slice.
					xImageSlice = xPadded[n, :, i * stride : i * stride + filterHeight, j * stride : j * stride + filterWidth]
					
					## Performing the dot product of the weight matrix with the image slice.
					outputActivationMap[n, k, i, j] = np.sum(xImageSlice * w[k]) + b[k]

	out = outputActivationMap

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, w, b, convParam)
	return out, cache


def convBackwardNaive(dOut, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dOut: Upstream derivatives.
	- cache: A tuple of (x, w, b, convParam) as in conv_forward_naive

	Returns a tuple of:
	- dx: Gradient with respect to x
	- dw: Gradient with respect to w
	- db: Gradient with respect to b
	"""
	dx, dw, db = None, None, None
						
	#############################################################################
	# TODO: Implement the convolutional backward pass.                          #
	#############################################################################
	
	## Unpacking cache.
	x, w, b, convParam = cache

	## Unpacking the convolutional parameters.
	stride, padWidth = convParam['stride'], convParam['pad']

	## Padding the input with zeroes.
	xPadded = np.pad(x, pad_width = ((0, 0), (0, 0), (padWidth, padWidth), (padWidth, padWidth)), mode = 'constant', constant_values = 0)
	
	## Defining the input size, depth, height and width.
	inputSize = x.shape[0]
	inputDepth = x.shape[1]
	inputHeight = x.shape[2]
	inputWidth = x.shape[3]

	## Defining the filter height, width, depth and number of filters.
	numFilters = w.shape[0]
	filerDepth = w.shape[1]
	filterHeight = w.shape[2]
	filterWidth = w.shape[3]

	## Defining the output size, depth, height and width.
	outputSize = x.shape[0]
	outputDepth = numFilters
	outputHeight = ((inputHeight - filterHeight + 2 * padWidth) / stride + 1)
	outputWidth = ((inputWidth - filterWidth + 2 * padWidth) / stride + 1)

	## Initializing the output activation map.
	outputActivationMap = np.empty([outputSize, outputDepth, outputHeight, outputWidth])

	## Create placeholders for the gradients.
	dW = np.zeros_like(w)
	dB = np.zeros_like(b)
	dX = np.zeros_like(x)
	dXPadded = np.zeros_like(xPadded)

	for n in range(0, inputSize):
		for k in range(0, numFilters):
			
			## Computing the gradient with respect to biases.
			dB[k] += dOut[n, k].sum()

			for i in range(0, outputHeight):
				for j in range(0, outputWidth):

					## Obtaining the relevant slice of the input.
					xImageSlice = xPadded[n, :, i * stride : i * stride + filterHeight, j * stride : j * stride + filterWidth]
					
					## Obtaining the upstream gradient of the corresponding activation map.
					dOutUpstream = dOut[n, k, i, j]
					
					## Computing the gradient with respect to the weights.
					dW[k] += xImageSlice * dOutUpstream
					
					## Computing the gradient with respect to the input.
					dXPadded[n, :, i * stride : i * stride + filterHeight, j * stride : j * stride + filterWidth] += w[k] * dOutUpstream

	## Extract dX from dXPadded.
	dX = dXPadded[:, :, padWidth : padWidth + inputHeight, padWidth : padWidth + inputWidth]

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dX, dW, dB


def maxPoolForwardNaive(x, poolParam):
	"""
	A naive implementation of the forward pass for a max pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- poolParam: dictionary with the following keys:
		- 'poolHeight': The height of each pooling region
		- 'poolWidth': The width of each pooling region
		- 'stride': The distance between adjacent pooling regions

	Returns a tuple of:
	- out: Output data
	- cache: (x, poolParam)
	"""
	out = None
	#############################################################################
	# TODO: Implement the max pooling forward pass                              #
	#############################################################################
	## Defining the input size, depth, height and width.
	inputSize = x.shape[0]
	inputDepth = x.shape[1]
	inputHeight = x.shape[2]
	inputWidth = x.shape[3]

	## Unpacking the pooling parameters.
	poolWidth, poolHeight, poolStride = poolParam['poolWidth'], poolParam['poolHeight'], poolParam['stride']

	## Defining the output size, depth, height and width.
	outputSize = x.shape[0]
	outputDepth = inputDepth
	outputHeight = ((inputHeight - poolHeight) / poolStride) + 1
	outputWidth = ((inputWidth - poolWidth) / poolStride) + 1

	## Initializing the output activation map.
	outputActivationMap = np.empty([outputSize, outputDepth, outputHeight, outputWidth])

	## Performing the pooling operation.
	for n in range(0, inputSize):
		for i in range(0, outputHeight):
			for j in range(0, outputWidth):
				
				## Obtaining the relevant slice along the depth.
				xImageSlice = x[n, :, i * poolStride : i * poolStride + poolHeight, j * poolStride : j * poolStride + poolWidth]
				
				## Filling in the correct values in the output placeholder along the depth.
				outputActivationMap[n, :, i, j] = np.amax(xImageSlice, axis = (-1, -2))

	out = outputActivationMap
				
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	cache = (x, poolParam)
	return out, cache


def maxPoolBackwardNaive(dOut, cache):
	"""
	A naive implementation of the backward pass for a max pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, poolParam) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None
	#############################################################################
	# TODO: Implement the max pooling backward pass                             #
	#############################################################################

	## Unpacking cache.
	x, poolParam = cache
	poolParam = {'poolHeight': 2, 'poolWidth': 2, 'stride': 2}

	## Create a placeholder for holding the gradient.
	dx = np.zeros_like(x)

	## Defining the input size, depth, height and width.
	inputSize = x.shape[0]
	inputDepth = x.shape[1]
	inputHeight = x.shape[2]
	inputWidth = x.shape[3]

	## Unpacking the pooling parameters.
	poolWidth, poolHeight, poolStride = poolParam['poolWidth'], poolParam['poolHeight'], poolParam['stride']

	## Defining the output size, depth, height and width.
	outputSize = x.shape[0]
	outputDepth = inputDepth
	outputHeight = ((inputHeight - poolHeight) / poolStride) + 1
	outputWidth = ((inputWidth - poolWidth) / poolStride) + 1
	
	for n in range(0, inputSize):
		for k in range(0, outputDepth):
			for i in range(0, outputHeight):
				for j in range(0, outputWidth):

					## Obtaining the relevant slice.
					xImageSlice = x[n, k, i * poolStride : i * poolStride + poolHeight, j * poolStride : j * poolStride + poolWidth] 
					
					## Obtaining the index of the maximum element in the above slice.
					maxElemIndex1, maxElemIndex2 = np.unravel_index(xImageSlice.argmax(), xImageSlice.shape)
				 
					## Computing the gradient.
					dx[n, k, i * poolStride : i * poolStride + poolHeight, j * poolStride : j * poolStride + poolWidth][maxElemIndex1, maxElemIndex2] = dOut[n, k, i, j]

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return dx


def spatialBatchnormForward(x, gamma, beta, bnParam):
	"""
	Computes the forward pass for spatial batch normalization.
	
	Inputs:
	- x: Input data of shape (N, C, H, W)
	- gamma: Scale parameter, of shape (C,)
	- beta: Shift parameter, of shape (C,)
	- bnParam: Dictionary with the following keys:
		- mode: 'train' or 'test'; required
		- eps: Constant for numeric stability
		- momentum: Constant for running mean / variance. momentum=0 means that
			old information is discarded completely at every time step, while
			momentum=1 means that new information is never incorporated. The
			default of momentum=0.9 should work well in most situations.
		- runningMean: Array of shape (D,) giving running mean of features
		- runningVar Array of shape (D,) giving running variance of features
		
	Returns a tuple of:
	- out: Output data, of shape (N, C, H, W)
	- cache: Values needed for the backward pass
	"""
	out, cache = None, None

	#############################################################################
	# TODO: Implement the forward pass for spatial batch normalization.         #
	#                                                                           #
	# HINT: You can implement spatial batch normalization using the vanilla     #
	# version of batch normalization defined above. Your implementation should  #
	# be very short; ours is less than five lines.                              #
	#############################################################################

	N, C, H, W = x.shape
	xReshaped = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
	out_tmp, cache = batchnorm_forward(xReshaped, gamma, beta, bnParam)
	out = out_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return out, cache


def spatialBatchnormBackward(dout, cache):
	"""
	Computes the backward pass for spatial batch normalization.
	
	Inputs:
	- dout: Upstream derivatives, of shape (N, C, H, W)
	- cache: Values from the forward pass
	
	Returns a tuple of:
	- dx: Gradient with respect to inputs, of shape (N, C, H, W)
	- dgamma: Gradient with respect to scale parameter, of shape (C,)
	- dbeta: Gradient with respect to shift parameter, of shape (C,)
	"""
	dx, dgamma, dbeta = None, None, None

	#############################################################################
	# TODO: Implement the backward pass for spatial batch normalization.        #
	#                                                                           #
	# HINT: You can implement spatial batch normalization using the vanilla     #
	# version of batch normalization defined above. Your implementation should  #
	# be very short; ours is less than five lines.                              #
	#############################################################################
	
	N, C, H, W = dout.shape
	dout_reshaped = dout.transpose(0,2,3,1).reshape(N*H*W, C)
	dx_tmp, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
	dx = dx_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return dx, dgamma, dbeta
	

def svmLoss(x, y):
	"""
	Computes the loss and gradient using for multiclass SVM classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = x.shape[0]
	correctClassScores = x[np.arange(N), y]
	margins = np.maximum(0, x - correctClassScores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N
	numPos = np.sum(margins > 0, axis = 1)
	dx = np.zeros_like(x)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= numPos
	dx /= N
	return loss, dx


def softmaxLoss(x, y):
	"""
	Computes the loss and gradient for softmax classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
		for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
		0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	probs = np.exp(x - np.max(x, axis = 1, keepdims = True))
	probs /= np.sum(probs, axis = 1, keepdims = True)
	N = x.shape[0]
	loss = -np.sum(np.log(probs[np.arange(N), y])) / N
	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx