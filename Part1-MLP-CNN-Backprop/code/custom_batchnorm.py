import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_neurons = n_neurons
    self.eps = eps
    self.gamma = nn.Parameter(torch.Tensor(n_neurons))
    self.beta = nn.Parameter(torch.Tensor(n_neurons))
    # Initialize learnable parameters
    self.gamma.data.fill_(1)
    self.beta.data.fill_(0)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if input.dim() != 2 or input.size(1) != self.n_neurons:
      raise ValueError
    x = input
    mean = torch.mean(x, dim=0)
    var = torch.var(x, dim=0, unbiased=False)
    x_mean = x-mean
    inv_var = 1/torch.sqrt(var+self.eps)
    norm_x = x_mean*inv_var
    out = self.gamma*norm_x + self.beta
    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = input
    mean = torch.mean(x, dim=0)
    var = torch.var(x, dim=0, unbiased=False)
    x_mean = x-mean
    inv_var = 1/torch.sqrt(var+eps)
    norm_x = x_mean*inv_var
    out = gamma*norm_x + beta
    ctx.constant=eps
    ctx.save_for_backward(norm_x, gamma, inv_var)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    norm_x, gamma, inv_var = ctx.saved_tensors 
    eps = ctx.constant
    B = norm_x.shape[0]
    grad_gamma = torch.sum(norm_x*grad_output, dim=0)
    grad_beta = torch.sum(grad_output, dim=0)
    grad_norm_x = grad_output * gamma
    grad_input = ((1./B)*inv_var) * (B*grad_norm_x - torch.sum(grad_norm_x, dim=0) - torch.sum(grad_norm_x*norm_x, dim=0)*norm_x)
    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_neurons = n_neurons
    self.eps = eps
    self.gamma = nn.Parameter(torch.Tensor(n_neurons))
    self.beta = nn.Parameter(torch.Tensor(n_neurons))
    # Initialize learnable parameters
    self.gamma.data.fill_(1)
    self.beta.data.fill_(0)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if input.dim() != 2 or input.size(1) != self.n_neurons:
      raise ValueError
    my_bn_fct = CustomBatchNormManualFunction()
    out = my_bn_fct.apply(input, self.gamma, self.beta, self.eps)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out






if __name__=='__main__':
    # create test batch
    n_batch = 128
    n_neurons = 4
    # create random tensor with variance 2 and mean 3
    x = 2*torch.randn(n_batch, n_neurons, requires_grad=True)+10
    print('Input data:\n\tmeans={}\n\tvars={}'.format(x.mean(dim=0).data, x.var(dim=0).data))

    # test CustomBatchNormAutograd
    print('3.1) Test automatic differentation version')
    bn_auto = CustomBatchNormAutograd(n_neurons)
    y_auto = bn_auto(x)
    print('\tmeans={}\n\tvars={}'.format(y_auto.mean(dim=0).data, y_auto.var(dim=0).data))

    # test CustomBatchNormManualFunction
    # this is recommended to be done in double precision
    print('3.2 b) Test functional version')
    input = x.double()
    gamma = torch.sqrt(10*torch.arange(n_neurons, dtype=torch.float64, requires_grad=True))
    beta = 100*torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
    bn_manual_fct = CustomBatchNormManualFunction(n_neurons)
    y_manual_fct = bn_manual_fct.apply(input, gamma, beta)
    print('\tmeans={}\n\tvars={}'.format(y_manual_fct.mean(dim=0).data, y_manual_fct.var(dim=0).data))
    # gradient check
    grad_correct = torch.autograd.gradcheck(bn_manual_fct.apply, (input,gamma,beta))
    if grad_correct:
        print('\tgradient check successful')
    else:
        raise ValueError('gradient check failed')

    # test CustomBatchNormManualModule
    print('3.2 c) Test module of functional version')
    bn_manual_mod = CustomBatchNormManualModule(n_neurons)
    y_manual_mod = bn_manual_mod(x)
    print('\tmeans={}\n\tvars={}'.format(y_manual_mod.mean(dim=0).data, y_manual_mod.var(dim=0).data))