"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    self.in_features = in_features
    self.out_features = out_features 
    self.params['weight'] = np.random.normal(0, 0.0001, (self.in_features, self.out_features))
    self.params['bias'] = np.zeros([self.out_features])
    self.grads['weight'] = np.zeros([self.in_features, self.out_features]) 
    self.grads['bias'] = np.zeros([self.out_features]) 
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    ####################### 
    # x is matrix of batch size x inf, W = ipf x outpf 
    out = np.dot(x, self.params['weight']) + self.params['bias'] 
    self.x_prev = x 
    self.out = out 
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = self.x_prev.T @ dout 
    self.grads['bias'] = np.sum(dout, axis=0) 
    dx = dout @ self.params['weight'].T 
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.neg_slope = neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(np.zeros(x.shape), x) + self.neg_slope*np.minimum(np.zeros(x.shape), x)
    self.x_prev = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.ones(self.x_prev.shape) 
    dx[self.x_prev < 0] = self.neg_slope 
    dx = np.multiply(dout, dx) 
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    max_x = np.max(x, axis=1)
    max_x = max_x[:, np.newaxis]
    max_x = np.repeat(max_x, x.shape[1], axis=1)
    exp = np.exp(x - max_x)
    out = np.divide(exp, np.sum(exp, axis=1)[:, np.newaxis]) 
    self.x_prev = x
    self.out_sm = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################  
    out_sm = self.out_sm
    out = out_sm.reshape(out_sm.shape[0], 1, out_sm.shape[1])
    out_T = out_sm.reshape(out_sm.shape[0], out_sm.shape[1], 1)
    diags = out_sm
    diags = diags[..., np.newaxis] * np.eye(diags.shape[-1])
    diags = np.array([np.diag(row) for row in out_sm])
    dx = diags - out_T @ out 
    dout = dout.reshape(dout.shape[0], 1, dout.shape[1])
    dx = dout @ dx
    dx = dx.reshape(dout.shape[0], dout.shape[2])
    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = -np.mean(np.log(x[np.where(y)]))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = -(y/x)*(1/y.shape[0]) 
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx