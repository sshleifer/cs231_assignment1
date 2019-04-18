from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    D, C = W.shape
    N, D = X.shape
    full_row_idx = np.arange(N)
    loss = 0.0
    dW = np.zeros_like(W)

    probas = np.zeros((N, C))
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****

    for i in range(N):
        for k in range(D):
            for j in range(C):
                probas[i, j] += X[i, k] * W[k, j]
        probas[i] = np.exp(probas[i] - max(probas[i]))
        probas[i] /= probas[i].sum()
        #feat_vals = X[i]
        #dW += feat_vals * probas[i]

    correct_class_probas = probas[full_row_idx, y]
    loss = -np.log(correct_class_probas).mean()
    loss += reg * (W**2).sum()

    # BACKWARD PASS
    probas[full_row_idx, y] -= 1
    for i in range(N):
        for k in range(D):
            feat_val = X[i,k]
            for j in range(C):
                yhat = probas[i, j]
                dW[k, j] += feat_val * yhat
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def stable_softmax(X):
    z = X - X.max(axis=1, keepdims=True)
    scores = np.exp(z)
    sm = scores / scores.sum(axis=1, keepdims=True)
    return sm


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    N = X.shape[0]
    full_row_idx = np.arange(N)
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    probas = stable_softmax(X @ W)

    correct_class_probas = probas[full_row_idx, y]
    loss = -np.log(correct_class_probas).mean()
    loss += reg * (W * W).sum()

    probas[full_row_idx, y] -= 1
    dW = (X.T.dot(probas) / N) + (2 * reg * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
