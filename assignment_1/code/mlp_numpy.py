"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.neg_slope = neg_slope
    self.layers = []
    self.leakyrelus = []
    # Logistic regression case
    if not self.n_hidden:
      self.outputlayer = LinearModule(self.n_inputs, self.n_classes)
      self.layers = []
      self.leakyrelus = []
    else:
      for layer, hidden_units in enumerate(self.n_hidden):
        linearlayer = LinearModule(n_inputs, hidden_units)
        leakyrelu = LeakyReLUModule(self.neg_slope)
        n_inputs = hidden_units
        self.layers.append(linearlayer)
        self.leakyrelus.append(leakyrelu)
      self.outputlayer = LinearModule(hidden_units, n_classes)
    self.softmax = SoftMaxModule()
    self.loss = CrossEntropyModule()
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x
    for layer, leakyrelu in zip(self.layers, self.leakyrelus):
        out = layer.forward(out)
        out = leakyrelu.forward(out)
    out = self.outputlayer.forward(out)
    out = self.softmax.forward(out) 
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout
    dx = self.softmax.backward(dx)
    dx = self.outputlayer.backward(dx)
    for layer, leakyrelu in zip(reversed(self.layers), reversed(self.leakyrelus)):
      dx = leakyrelu.backward(dx)
      dx = layer.backward(dx)
    ########################
    # END OF YOUR CODE    #
    #######################

    return
