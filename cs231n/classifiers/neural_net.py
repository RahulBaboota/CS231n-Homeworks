import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, inputSize, hiddenSize, outputSize, std = 1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - inputSize: The dimension D of the input data.
    - hiddenSize: The number of neurons H in the hidden layer.
    - outputSize: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(inputSize, hiddenSize)
    self.params['b1'] = np.zeros(hiddenSize)
    self.params['W2'] = std * np.random.randn(hiddenSize, outputSize)
    self.params['b2'] = np.zeros(outputSize)

  def loss(self, X, y = None, reg = 0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    ## Computing the first hidden layer.
    hiddenLayer1 = X.dot(W1) + b1

    ## Applying Relu to the hidden layer.
    activatedHiddenLayer1 = np.clip(hiddenLayer1, 0, None)

    ## Computing the second hidden layer.
    hiddenLayer2 = activatedHiddenLayer1.dot(W2) + b2

    ## Storing this matrix in the scores variable.
    scores = hiddenLayer2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    
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
    loss = loss/N

    ## Add regularisation loss.
    loss = loss + 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    dO = softmaxScores

    ## Computing dL/dO (Softmax Gradient).
    dO[np.arange(N), y] -= 1
    dO /= N

    ## Computing dL/db2.
    grads['b2'] = np.sum(dO, axis = 0)

    ## Computing dL/dW2.
    grads['W2']= activatedHiddenLayer1.T.dot(dO) + reg * W2  

    ## Computing dL/dActivatedHiddenLayer1.
    dActivatedHiddenLayer1 = dO.dot(W2.T)

    ## Computing dL/dHiddenLayer1 (Backprop through Relu).
    dActivatedHiddenLayer1[activatedHiddenLayer1 <= 0] = 0

    ## Computing dL/db1.
    grads['b1'] = np.sum(dActivatedHiddenLayer1, axis = 0)

    ## Computing dL/dW1.
    grads['W1'] = X.T.dot(dActivatedHiddenLayer1) + reg * W1
    

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, XVal, yVal,
            learningRate = 1e-3, learningRateDecay = 0.95,
            reg = 1e-5, numIters = 100,
            batchSize = 200, verbose = False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - XVal: A numpy array of shape (N_val, D) giving validation data.
    - yVal: A numpy array of shape (N_val,) giving validation labels.
    - learningRate: Scalar giving learning rate for optimization.
    - learningRateDecay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - numIters: Number of steps to take when optimizing.
    - batchSize: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    numTrain = X.shape[0]
    iterations_per_epoch = max(numTrain / batchSize, 1)

    # Use SGD to optimize the parameters in self.model
    lossHistory = []
    trainAccHistory = []
    valAccHistory = []

    for it in xrange(numIters):

      XBatch = None
      yBatch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in XBatch and yBatch respectively.                             #
      #########################################################################
      ## Creating an array which randomly selects images.
      randomIndices = np.random.choice(np.arange(numTrain), size = batchSize)
      XBatch = X[randomIndices]
      yBatch = y[randomIndices]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(XBatch, y = yBatch, reg = reg)
      lossHistory.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      
      ## Updating the weights and biases using stochastic gradient descent.

      self.params['W2'] -= learningRate * grads['W2']
      self.params['b2'] -= learningRate * grads['b2']

      self.params['W1'] -= learningRate * grads['W1']
      self.params['b1'] -= learningRate * grads['b1']

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, numIters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(XBatch) == yBatch).mean()
        val_acc = (self.predict(XVal) == yVal).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learningRate *= learningRateDecay

    return {
      'lossHistory': lossHistory,
      'trainAccHistory': trainAccHistory,
      'valAccHistory': valAccHistory,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - yPred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, yPred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    yPred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return yPred


