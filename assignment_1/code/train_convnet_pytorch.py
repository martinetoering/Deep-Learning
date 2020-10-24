"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  # List of examples that are correct
  correct = np.where(pred == targets)[0]
  n_correct = len(correct)
  accuracy = n_correct / predictions.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # Load cifar data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  learning_rate = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  max_steps = FLAGS.max_steps
  eval_freq = FLAGS.eval_freq

  # Manual seed
  torch.manual_seed(1)

  # Initialize model
  model = ConvNet(3, 10) 
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  train = cifar10['train']
  val = cifar10['validation']
  test = cifar10['test']

  n_examples = train.num_examples
  print("Number of training examples:", n_examples)
  print("Train")
  training_accuracy = []
  training_loss = []
  test_accuracy = []

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()
  n_iterations = 0
  while n_iterations < max_steps:
    model = model.train()
    if (n_iterations+1) % eval_freq == 0: 
      print("Iteration {}".format(n_iterations+1))
    # Get one batch
    x, y = train.next_batch(batch_size)
    optimizer.zero_grad()
    x = torch.from_numpy(x).to(device)
    y = np.argmax(y, axis=1) # From one-hot matrix to vector
    y = torch.from_numpy(y).to(device)
    out = model(x)
    acc = accuracy(out.data.cpu().numpy(), y.data.cpu().numpy())
    loss = criterion(out, y)
    # Store loss and accuracy
    training_loss.append(loss.cpu().item())
    training_accuracy.append(acc)
    loss.backward()
    optimizer.step()
    if (n_iterations+1) % eval_freq  == 0:
      print("Training accuracy:", acc)
      model = model.eval()
      u_outs = np.empty([test.num_examples, 10])
      u, v = test.next_batch(test.num_examples)
      current_batch = 0
      for i in range(test.num_examples//batch_size+1):
        a = u[current_batch:current_batch+batch_size, :, :, :]
        a = torch.from_numpy(a).to(device)
        a_out = model.forward(a)
        a_out = a_out.data.cpu().numpy()
        u_outs[current_batch:current_batch+batch_size] = a_out
        current_batch += batch_size
      v = np.argmax(v, axis=1) # From one-hot matrix to vector
      acc_test = accuracy(u_outs, v)
      test_accuracy.append(acc_test)
      print("Test accuracy:", acc_test)
    n_iterations += 1

  with open("pytorch_convnet_results.txt", "w")  as f:
    f.write("PyTorch ConvNet: Train acc, train loss and test acc\n")
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
  FLAGS, unparsed = parser.parse_known_args()

  main()