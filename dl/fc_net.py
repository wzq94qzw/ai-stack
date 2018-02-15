import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

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

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT batch normalize the output scores.
    # ================================================================ #
    
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    self.params['b1'] = np.zeros(hidden_dims[0])

    for i in np.arange(1, self.num_layers-1):
      self.params['W' + str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])  
      self.params['b' + str(i+1)] = np.zeros(hidden_dims[i])    
      
    self.params['W' + str(len(hidden_dims)+1) ] = weight_scale * np.random.randn(hidden_dims[len(hidden_dims)-1], num_classes) 
    self.params['b' + str(len(hidden_dims)+1) ] = np.zeros(num_classes)

    if self.use_batchnorm:
      # self.params['gamma1'] = np.ones(hidden_dims[0])
      # self.params['beta1'] = np.zeros(hidden_dims[0])
      for i in np.arange(self.num_layers-1):
        self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
        self.params['beta'  + str(i+1)] = np.zeros(hidden_dims[i])

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
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
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
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
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #

    caches = []
    caches_do = []
    h_tmp = X

    for i in np.arange(1, self.num_layers):
        if self.use_batchnorm:
          h_tmp, cache_tmp = affine_batchnorm_relu_forward(
            h_tmp, self.params['W' + str(i)], self.params['b' + str(i)], 
            self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i-1])
        else:
          h_tmp, cache_tmp = affine_relu_forward(h_tmp, self.params['W' + str(i)], self.params['b' + str(i)])
        

        if self.use_dropout:
          h_tmp, cache_do = dropout_forward(h_tmp, self.dropout_param)
          caches_do.append(cache_do)

        caches.append( (h_tmp, cache_tmp ))

    h_L, cache_h_L = affine_forward(h_tmp, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
    caches.append( (h_L, cache_h_L) )
    scores = h_L

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    #
    #   BATCHNORM: Incorporate the backward pass of the batchnorm.
    #
    #   DROPOUT: Incorporate the backward pass of dropout.
    # ================================================================ #

    loss, d_L = softmax_loss(caches[-1][0], y)

    for i in np.arange(1, self.num_layers+1):
      loss += 0.5*self.reg*(np.sum(np.square(self.params['W' + str(i)])))

    dh_L = affine_backward(d_L, caches[-1][1])
    grads['W' + str(self.num_layers)] = dh_L[1] + self.reg * self.params['W' + str(self.num_layers)]
    grads['b' + str(self.num_layers)] = dh_L[2]

    dh_tmp = dh_L

    for i in np.arange(self.num_layers-2, -1, -1):

      d_tmp = dh_tmp[0]
      if self.use_dropout:
        d_tmp = dropout_backward(dh_tmp[0], caches_do[i])

      if self.use_batchnorm:
        dh_tmp = affine_batchnorm_relu_backward(d_tmp, caches[i][1])
        grads['gamma' + str(i+1)] = dh_tmp[3]
        grads['beta' + str(i+1)] = dh_tmp[4]
      else:
        dh_tmp = affine_relu_backward(d_tmp, caches[i][1])

      grads['W' + str(i+1)] = dh_tmp[1] + self.reg * self.params['W' + str(i+1)]
      grads['b' + str(i+1)] = dh_tmp[2]


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads
