"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import torch.nn as nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
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
    super().__init__()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.neg_slope = neg_slope
    self.layers = nn.ModuleList() # Preserve parameters
    self.leakyrelus = nn.ModuleList()
    # Logistic regression case
    if not self.n_hidden:
      self.outputlayer = nn.Linear(self.n_inputs, self.n_classes)
      self.layers = []
      self.leakyrelus = []
    else:
      for layer, hidden_units in enumerate(self.n_hidden):
        linearlayer = nn.Linear(n_inputs, hidden_units)
        leakyrelu = nn.LeakyReLU(negative_slope=neg_slope)
        n_inputs = hidden_units
        self.layers.append(linearlayer)
        self.leakyrelus.append(leakyrelu)
      self.outputlayer = nn.Linear(hidden_units, n_classes)

    # Weight initialization
    # for layer in self.layers:
    #   nn.init.normal_(layer.weight, mean=0, std=0.0001) # Initialize same way as in mlp numpy
    #   nn.init.constant_(layer.bias, 0)
    # nn.init.normal_(self.outputlayer.weight, mean=0, std=0.0001)
    # nn.init.constant_(self.outputlayer.bias, 0)

    # Weight initialization
    for layer in self.layers:
      nn.init.kaiming_normal_(layer.weight)
      nn.init.constant_(layer.bias, 0)
    nn.init.kaiming_normal_(self.outputlayer.weight)
    nn.init.constant_(self.outputlayer.bias, 0)
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
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
