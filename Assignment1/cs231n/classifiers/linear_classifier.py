import numpy as np
from cs231n.classifiers.softmax import *
from cs231n.classifiers.linear_svm import *

class LinearClassifier(object):

  def __init__(self):

    self.W = None

  def train(self, X, y, learningRate = 1e-3, reg = 1e-5, numIters = 100, batchSize = 200, verbose = False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learningRate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - numIters: (integer) number of steps to take when optimizing
    - batchSize: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """

    ## Number of training samples and the dimensions of each training sample.
    numTrain, dim = X.shape


    ## Total number of classes (assume y takes values 0...K-1 where K is number of classes)
    numClasses = np.max(y) + 1 

    if self.W is None:

      ## Initialise W randomly.
      self.W = 0.001 * np.random.randn(dim, numClasses)

    # Run stochastic gradient descent to optimize W
    lossHistory = []

    for it in xrange(numIters):

      XBatch = None
      yBatch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batchSize elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in XBatch and their corresponding labels in           #
      # yBatch; after sampling XBatch should have shape (dim, batchSize)   #
      # and yBatch should have shape (batchSize,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      
      ## Creating an array which randomly selects images.
      randomIndices = np.random.choice(np.arange(numTrain), size = batchSize)
      XBatch = X[randomIndices]
      yBatch = y[randomIndices]


      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(XBatch, yBatch, reg)
      lossHistory.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      
      ## Updating the weights using stochastic gradient descent.
      self.W -= learningRate * grad

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, numIters, loss)

    return lossHistory

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - yPred: Predicted labels for the data in X. yPred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    yPred = np.zeros(X.shape[1])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in yPred.            #
    ###########################################################################
    
    ## Performing the forward pass to compute the raw scores.
    rawScores = X.dot(self.W)

    ## Finding the prediction made by the classifier.
    yPred = rawScores.argmax(axis = 1)


    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return yPred
  
  def loss(self, XBatch, yBatch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - XBatch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - yBatch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, XBatch, yBatch, reg):
    return svm_loss_vectorized(self.W, XBatch, yBatch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, XBatch, yBatch, reg):
    return softmax_loss_vectorized(self.W, XBatch, yBatch, reg)

