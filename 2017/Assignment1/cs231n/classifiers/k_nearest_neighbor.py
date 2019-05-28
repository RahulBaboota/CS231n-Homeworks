import numpy as np
from collections import Counter
from past.builtins import xrange

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (numTrain, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
       y[i] is the label for X[i].
    """

    self.XTrain = X
    self.yTrain = y

  def predict(self, X, k = 1, numLoops = 0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (numTest, D) containing test data consisting
       of numTest samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - numLoops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (numTest,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """

    if numLoops == 0:
      distMat = self.computeDistancesTwoLoops(X)

    elif numLoops == 1:
      distMat = self.computeDistancesOneLoop(X)

    elif numLoops == 2:
      distMat = self.computeDistancesTwoLoop(X)

    else:
      raise ValueError('Invalid value %d for numLoops' % numLoops)

    return self.predictLabels(distMat, k = k)

  def computeDistancesTwoLoops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.XTrain using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (numTest, D) containing test data.

    Returns:
    - distMat: A numpy array of shape (numTest, numTrain) where distMat[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """

    ## Computing the shape of the test and train data.
    numTest = X.shape[0]
    numTrain = self.XTrain.shape[0]

    ## Initializing the distance matrix.
    distMat = np.zeros((numTest, numTrain))

    #####################################################################
    # TODO:                                                             #
    # Compute the l2 distance between the ith test point and the jth    #
    # training point, and store the result in distMat[i, j]. You should   #
    # not use a loop over dimension.                                    #
    #####################################################################
        
    for i in xrange(numTest):
      for j in xrange(numTrain):

        ## Computing the l2 distance between the training and test image.
        distMat[i, j] = np.linalg.norm(self.XTrain[j] - X[i])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
      
    return distMat

  def computeDistancesOneLoop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.XTrain using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """

    ## Computing the shape of the test and train data.
    numTest = X.shape[0]
    numTrain = self.XTrain.shape[0]

    ## Initializing the distance matrix.
    distMat = np.zeros((numTest, numTrain))

    #######################################################################
    # TODO:                                                               #
    # Compute the l2 distance between the ith test point and all training #
    # points, and store the result in distMat[i, :].                        #
    #######################################################################

    for i in xrange(numTest):
      
      ## Computing the l2 distance between the training and test image.
      distMat[i, :] = np.linalg.norm((self.XTrain - X[i]), axis = -1)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return distMat

  def computeDistancesNoLoops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.XTrain using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    ## Computing the shape of the test and train data.
    numTest = X.shape[0]
    numTrain = self.XTrain.shape[0]

    ## Initializing the distance matrix.
    distMat = np.zeros((numTest, numTrain))

    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # distMat.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    
    distMat = np.sqrt(np.sum(X ** 2, axis = 1).reshape(numTest, 1) + np.sum(self.XTrain ** 2, axis = 1) - 2 * X.dot(self.XTrain.T))

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################

    return distMat

  def predictLabels(self, distMat, k = 1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - distMat: A numpy array of shape (numTest, numTrain) where distMat[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (numTest,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """

    ## Number of testing images.
    numTest = distMat.shape[0]

    ## Initializing prediction vector.
    yPred = np.zeros(numTest)

    for i in xrange(numTest):

      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closestY = []

      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.yTrain to find the labels of these       #
      # neighbors. Store these labels in closestY.                            #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      ## Finding the top K indices based on the distance value.
      topKIndx = np.argsort(distMat[i])[:k]

      ## Storing the labels of the K Nearest Neighbours.
      closestY = self.yTrain[topKIndx]

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closestY of labels.    #
      # Store this label in yPred[i]. Break ties by choosing the smaller      #
      # label.                                                                #
      #########################################################################

      ## Compute the occurences of all the nearest neighbours.
      countOccurence = Counter(closestY)

      ## Choose the label based on majority voting.
      countMax = countOccurence.most_common()

      yPred[i] = countMax[0][0]

      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return yPred