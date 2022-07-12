"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  batch_size = predictions[0]
  # Top 1 predictions
  pred = np.argmax(predictions, axis=1)
  # Target classes
  tar = np.argmax(targets, axis=1)
  # List of examples that are correct
  correct = np.where(pred == tar)[0]
  n_correct = len(correct)
  accuracy = n_correct / predictions.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # Load cifar data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  learning_rate = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  max_steps = FLAGS.max_steps
  eval_freq = FLAGS.eval_freq

  # Initialize model
  model = MLP(32*32*3, dnn_hidden_units, 10, neg_slope) 
  train = cifar10['train']
  val = cifar10['validation']
  test = cifar10['test']

  n_examples = train.num_examples
  print("Number of training examples:", n_examples)
  print("Train")
  training_accuracy = []
  training_loss = []
  test_accuracy = []

  n_iterations = 0
  while n_iterations < max_steps:
    if (n_iterations+1) % eval_freq == 0: 
      print("Iteration {}".format(n_iterations+1))
    # Get one batch
    x, y = train.next_batch(batch_size)
    x = x.reshape(batch_size, 32*32*3) 
    out = model.forward(x)
    acc = accuracy(out, y)
    loss = model.loss.forward(out, y)
    # Store loss and accuracy
    training_loss.append(loss)
    training_accuracy.append(acc)
    dout = model.loss.backward(out, y)
    model.backward(dout)
    if (n_iterations+1) % eval_freq == 0:
      print("Training accuracy:", acc)
      u, v = test.images, test.labels
      u = u.reshape(u.shape[0], 32*32*3) 
      u_out = model.forward(u)
      acc_test = accuracy(u_out, v)
      test_accuracy.append(acc_test)
      print("Test accuracy at iteration {}:".format(n_iterations+1), acc_test)
    # Stochastic gradient descent
    for layer in model.layers:
      layer.params['weight'] -= learning_rate * layer.grads['weight'] 
      layer.params['bias'] -= learning_rate * layer.grads['bias'] 
    model.outputlayer.params['weight'] -= learning_rate * model.outputlayer.grads['weight'] 
    model.outputlayer.params['bias'] -= learning_rate * model.outputlayer.grads['bias'] 
    n_iterations += 1

  with open("numpy_results.txt", "w")  as f:
    f.write("NumPy MLP: Train acc, train loss and test acc\n")
    f.write("{}\n".format(training_accuracy))
    f.write("{}\n".format(training_loss))
    f.write("{}\n".format(test_accuracy))
    f.close()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()