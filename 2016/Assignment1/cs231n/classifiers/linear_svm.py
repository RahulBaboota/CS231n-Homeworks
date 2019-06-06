import numpy as np
from random import shuffle

def svmLossNaive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
	that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""

	## initialize the gradient as zero.
	dW = np.zeros(W.shape) 

	## Computing the number of classes.
	numClasses = W.shape[1]

	## Computing the minibatch size.
	numTrain = X.shape[0]

	## Initialising loss.
	loss = 0.0

	for i in xrange(numTrain):

		## Computing the raw scores for each image.
		scores = X[i].dot(W)

		## Computing the score for the correct class.
		correctClassScore = scores[y[i]]

		for j in xrange(numClasses):

			## Not computing the margin for the correct label.
			if j == y[i]:
				continue

			## Computing the margin for the particular class.
			margin = scores[j] - correctClassScore + 1 

			## Threshholding the margin at zero.
			if margin > 0:
				loss += margin

				## Computing the gradient for the non-correct class.
				dW[:, j] += X[i]

				## Computing the gradient for the correct class.
				dW[:, y[i]] -= X[i]

	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by numTrain.
	loss /= numTrain
	dW /= numTrain

	## Compute the regularization gradient.
	dW += 2 * reg * W

	# Add regularization to the loss.
	loss += 0.5 * reg * np.sum(W * W)

	#############################################################################
	# TODO:                                                                     #
	# Compute the gradient of the loss function and store it dW.                #
	# Rather that first computing the loss and then computing the derivative,   #
	# it may be simpler to compute the derivative at the same time that the     #
	# loss is being computed. As a result you may need to modify some of the    #
	# code above to compute the gradient.                                       #
	#############################################################################

	return loss, dW


def svmLossVectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""

	## Initialising loss.
	loss = 0.0

	## initialize the gradient as zero.
	dW = np.zeros(W.shape) 

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	#############################################################################

	## Computing the minibatch size.
	numTrain = X.shape[0]

	## Computing the raw scores for each image.
	scores = X.dot(W)

	## Creating an indexing mask.
	mask = np.arange(numTrain)

	## Computing the correct class scores.
	correctClassScores = scores[mask, y][:, np.newaxis]

	## Computing the margins.
	margins = np.maximum(0, scores - correctClassScores + 1)
	margins[mask, y] = 0

	## Summing up the margins for all images.
	loss = np.sum(margins)

	## Normalising the loss.
	loss /= numTrain

	# Add regularization to the loss.
	loss += 0.5 * reg * np.sum(W ** 2)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################


	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	#############################################################################

	## Creating a mask for binary setting of scores greater than and less than 0.
	marginCounts = np.zeros(margins.shape)
	marginCounts[margins > 0] = 1
	marginCounts[mask, y] = -np.sum(margins > 0, axis = 1)

	dW = X.T.dot(marginCounts)

	## Normalise the gradient.
	dW /= numTrain

	## Compute the regularization gradient.
	dW += 2 * reg * W
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return loss, dW