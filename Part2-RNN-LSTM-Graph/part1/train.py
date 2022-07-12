################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append(".")

import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

import csv
import os 
from part1.plot_part1 import plot_train_part1
from part1.plot_part1 import plot_test_part1


# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

	assert config.model_type in ('RNN', 'LSTM')

	# Initialize the device which to run the model on
	device = torch.device(config.device)
	print("device:", device)

	# Initialize the model that we are going to use
	if config.model_type == 'RNN':
		print("RNN")
		model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)
	if config.model_type == 'LSTM':
		print("LSTM")
		model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)
	model = model.to(device)
	
	torch.manual_seed(config.manual_seed)
	
	# Results and logging
	results_folder = os.path.join(os.getcwd(), '{}_{}'.format(config.results_folder, config.model_type))
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print("Created dir: {}".format(results_folder))

	file_name = "{}_palindrome_{}_exp_{}".format(config.model_type, config.input_length+1, config.manual_seed)
	config.log_file = os.path.join(results_folder, "train_{}.log".format(file_name))
	if not config.test_log_file:
		config.test_log_file = os.path.join(results_folder, "test_{}.log".format(config.model_type))
	log_file = open(config.log_file, "w+")
	logger = csv.writer(log_file, delimiter='\t')
	logger.writerow(["step", "loss", "accuracy"])
	if os.path.exists(config.test_log_file):
		file_mode = "a"
	else:
		file_mode = "w"
	test_log_file = open(config.test_log_file, file_mode)
	test_logger = csv.writer(test_log_file, delimiter='\t')
	if file_mode == "w":
		test_logger.writerow(["input_length", "exp_num", "accuracy"])

	# Store config 
	if config.test_mode is False:
		config_file_name = os.path.join(results_folder, "config_{}.txt".format(file_name))
	else:
		config_file_name = os.path.join(results_folder, "config_{}.txt".format(config.model_type))
	if not os.path.exists(config_file_name):
		with open(config_file_name, 'w') as config_file:
			print(vars(config), file=config_file)

	# Initialize the dataset and data loader (note the +1)
	dataset = PalindromeDataset(config.input_length+1)
	data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

	# Separate dataloader for test (not strictly necessary)
	test_data = PalindromeDataset(config.input_length+1)
	test_loader = DataLoader(dataset, config.batch_size, num_workers=1)

	# Setup the loss and optimizer
	criterion = nn.CrossEntropyLoss().to(device)
	print("Learning rate:", config.learning_rate)
	optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

	train_loss = []
	train_accuracy = []

	for step, (batch_inputs, batch_targets) in enumerate(data_loader):

		# Only for time measurement of step through network
		t1 = time.time()

		model.train()

		optimizer.zero_grad()
		# Batch inputs to one-hot
		onehot_batch_inputs = torch.eye(10)[batch_inputs.long()]
		batch_inputs = onehot_batch_inputs.to(device)
		out = model(batch_inputs)

		############################################################################
		# QUESTION: what happens here and why? Gradients are clipped or scaled 
		# to avoid exploding gradients 
		############################################################################
		torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
		############################################################################

		batch_targets = batch_targets.to(device)
		loss = criterion(out, batch_targets)
		num_correct = (out.argmax(dim=1) == batch_targets).float().cpu()
		accuracy = torch.mean(num_correct, dim=0).item()
		loss.backward(retain_graph=True)
		optimizer.step()
		train_loss.append(loss.cpu().item())
		train_accuracy.append(accuracy)
		
		# Just for time measurement
		t2 = time.time()
		examples_per_second = config.batch_size/float(t2-t1)

		if step % 10 == 0:

			print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
				  "Accuracy = {:.2f}, Loss = {:.3f}, Palindrome length = {}".format(
					datetime.now().strftime("%Y-%m-%d %H:%M"), step,
					config.train_steps, config.batch_size, examples_per_second,
					accuracy, loss, config.input_length+1
			))

			logger.writerow([step, loss.cpu().item(), accuracy])
			log_file.flush()
			# If train convergence set to true we allow early stopping if loss does not decrease
			if config.train_convergence:
				if step > config.steps_convergence_train:
					if min(train_loss) < min(train_loss[step-config.steps_convergence_train:step]):
						print("Convergence at {}, stopping train".format(step))
						break

		if step == config.train_steps:
			# If you receive a PyTorch data-loader error, check this bug report:
			# https://github.com/pytorch/pytorch/pull/9655
			break

	print('Done training.')

	if config.do_test is True:

		print('Evaluation (Test)')
		model = model.eval()

		test_accuracy = []

		for test_step, (batch_inputs, batch_targets) in enumerate(test_loader):

			onehot_batch_inputs = torch.eye(10)[batch_inputs.long()]
			onehot_batch_targets = torch.eye(10)[batch_targets.long()]
			batch_inputs = onehot_batch_inputs.to(device)
			out = model(batch_inputs)
			batch_targets = batch_targets.to(device)
			loss = criterion(out, batch_targets)
			num_correct = (out.argmax(dim=1) == batch_targets).float().cpu()
			accuracy = torch.mean(num_correct, dim=0).item()
			test_accuracy.append(accuracy)

			if test_step % 10 == 0:

				print("[{}] Test Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
					  "Accuracy = {:.2f}, Loss = {:.3f}, Palindrome length = {}".format(
						datetime.now().strftime("%Y-%m-%d %H:%M"), test_step,
						config.test_steps, config.batch_size, examples_per_second,
						accuracy, loss, config.input_length+1
				))

			if test_step == config.test_steps:
				test_logger.writerow([config.input_length, config.manual_seed, sum(test_accuracy) / len(test_accuracy)])
				test_log_file.flush()
				break

	return config, results_folder

 ################################################################################
 ################################################################################

if __name__ == "__main__":

	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Model params
	parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
	parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
	parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
	parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
	parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
	parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
	parser.add_argument('--max_norm', type=float, default=10.0)
	parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

	parser.add_argument('--results_folder', type=str, default='sample_folder', help="Where to store logs and/or plots")
	parser.add_argument('--log_file', type=str, help="Train log file")
	parser.add_argument('--manual_seed', type=int, default=1, help="Torch manual seed")
	parser.add_argument('--train_convergence', type=bool, default=True, help="Early stopping is possible in train")
	parser.add_argument('--steps_convergence_train', type=int, default=1000, 
		help="How many train steps the loss should not decrease for convergence")
	parser.add_argument('--do_test', type=bool, default=False, help="If test should be performed")
	parser.add_argument('--test_mode', type=bool, default=False, help="Do train and test for all input lengths")
	parser.add_argument('--test_steps', type=int, default=5000, help="Number of test steps")
	parser.add_argument('--test_log_file', type=str, help="Test log file")
	config = parser.parse_args()

	# Test mode 
	if config.test_mode:
		config.do_test = True
		# Test the following palindrome lengths
		step_sizes = [range(10, 20, 10), range(20, 40, 2), range(40, 81, 4)]
		print("We are doing these palindrome lengths:", [list(i) for i in step_sizes])
		for steps in step_sizes:
			for input_length in steps:
				config.input_length = input_length-1
				print("Test palindrome length: {}".format(config.input_length+1))
				# We do 3 different seeds
				for i in range(3):
					seed = 0
					config.manual_seed = seed
					print("Experiment number {}".format(seed))
					config, results_folder = train(config)
					# Plot train only for first experiment
					# Uncomment to plot 
					# if i == 0:	
						# plot_train_part1(config.model_type, results_folder, config.log_file, config.input_length)
		# Uncomment to plot
		# plot_test_part1(config.model_type, results_folder, config.test_log_file)
 
	else:   
		# Train the model
		config, results_folder = train(config)
		# Uncomment to plot
		# plot_train_part1(config.model_type, results_folder, config.log_file, config.input_length)