import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    ## Computing the raw scores or the logits.
    scores = X.dot(W)
    
    ## Computing number of training instances.
    numTrain = X.shape[0]

    for i in range(0, numTrain):

        ## Accessing the scores array for each image.
        imgScoreMat = scores[i]

        ## Finding the maximum class score in above array.
        maxClsScore = np.max(imgScoreMat)

        ## Normalise the raw scores to avoid exponential score blow-up.
        ## To do so, subtract the maximum score from each score value for each image.
        normScoreMat = imgScoreMat - maxClsScore

        ## Exponentiate the normalised class scores.
        expScoreMat = np.exp(normScoreMat)

        ## Computing the sum of all the exponentiated scores.
        expScoresSum = np.sum(expScoreMat, axis = 0, keepdims = True)
            
        ## Compute the probabilities (or softmax scores) of each class.
        imgSoftmaxScores = expScoreMat/expScoresSum
        
        ## Compute the probabilities (or softmax scores) of each class.
        imgSoftmaxScores = expScoreMat/expScoresSum

        ## Finding the softmax score for the correct class.
        corrSoftScore = imgSoftmaxScores[y[i]]

        ## Computing the loss for the particular image and appending to the running loss.
        loss = loss + -np.log(corrSoftScore/np.sum(imgSoftmaxScores))
        
        ## Updating the gradient Matrix.
        dW[i] = imgSoftmaxScores - y[i]
              
    
    ## Compute the full training loss by dividing the cummulative loss by the number of training instances.
    loss = loss/numTrain
    
    ## Add regularisation loss.
    loss = loss + 0.5 * reg * np.sum(W*W) 
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    ## Defining the number of training instances in the current minibatch.
    numTrain = X.shape[0]
    
    ## Computing the raw scores or the logits.
    scores = X.dot(W)
    
    ## Normalise the raw scores to avoid exponential score blow-up.
    ## To do so, subtract the maximum score from each score value for each image.
    expScores = np.exp(scores - np.max(scores, axis = 1, keepdims = True))
    
    ## Compute the probabilities (or softmax scores) of each class.
    softmaxScores = expScores/np.sum(expScores, axis = 1, keepdims = True)
    
    ## Creating a 1-D matrix containing the softmax score of the correct class.
    corrSoftScore = np.choose(y, softmaxScores.T)
    
    ## Computing the cross-entropy loss.
    loss = -np.sum(np.log(corrSoftScore), axis = 0, keepdims = True)
    
    ## Extracting the single float value from the 1 element numpy array.
    loss = loss[0]
    
    ## Compute the full training loss by dividing the cummulative loss by the number of training instances.
    loss = loss/numTrain
    
    ## Add regularisation loss.
    loss = loss + 0.5*reg*np.sum(W*W) 
    
    ## Computing the gradient with the help of chain rule.
    dO = softmaxScores - y[:, None]
    dW = X.T.dot(dO)
    dW /= numTrain
    
    ## Gradient Regularisation.
    dW = dW + reg*W
    
    return loss, dW

