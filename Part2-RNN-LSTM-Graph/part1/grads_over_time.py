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
sys.path.append('..')

import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

import csv
import os 
from part1.plot_part1 import plot_grads_timesteps

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################


def obtain_grads_over_time(config, exp_num):

	assert config.model_type in ('RNN', 'LSTM')

	# Initialize the device which to run the model on
	device = torch.device(config.device)

	# Initialize the model that we are going to use
	if config.model_type == 'RNN':
		model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,
		 	config.num_classes, config.batch_size, device)
	if config.model_type == 'LSTM':
		model = LSTM(config.input_length, config.input_dim, config.num_hidden, 
			config.num_classes, config.batch_size, device)
	model = model.to(device)
	
	torch.manual_seed(config.manual_seed)
	
	# Results and logging
	results_folder = os.path.join(os.getcwd(), '{}_{}'.format(config.results_folder, config.input_length))
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print("Created dir: {}".format(results_folder))

	config.log_file = os.path.join(results_folder, "gradsovertime_{}.log".format(config.model_type))
	if os.path.exists(config.log_file):
		file_mode = "a"
	else:
		file_mode = "w"
	log_file = open(config.log_file, file_mode)
	logger = csv.writer(log_file, delimiter='\t')
	if file_mode == "w":
		logger.writerow(["exp_num", "input_length", "time_step", "norm_grad"])

	# Store config 
	config_file_name = os.path.join(results_folder, "config_gradsovertime_{}.txt".format(config.model_type))
	if not os.path.exists(config_file_name):
		with open(config_file_name, 'w') as config_file:
			print(vars(config), file=config_file)

	# Initialize the dataset and data loader (note the +1)
	dataset = PalindromeDataset(config.input_length+1)
	data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

	# Setup the loss and optimizer
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

	for step, (batch_inputs, batch_targets) in enumerate(data_loader):

		# Only for time measurement of step through network
		t1 = time.time()

		model.train()

		# Batch inputs to one-hot
		onehot_batch_inputs = torch.eye(10)[batch_inputs.long()]
		batch_inputs = onehot_batch_inputs.to(device)

		hs = []
		x = batch_inputs

		if config.model_type == 'LSTM':
			h_prev = model.h_t
			c_prev = model.c_t
			for t in range(model.seq_length):
				g_t = torch.tanh(x[:, t, :] @ model.w_gx + h_prev @ model.w_gh + model.b_g)
				i_t = torch.sigmoid(x[:, t, :] @ model.w_ix + h_prev @ model.w_ih + model.b_i)
				f_t = torch.sigmoid(x[:, t, :] @ model.w_fx + h_prev @ model.w_fh + model.b_f)
				o_t = torch.sigmoid(x[:, t, :] @ model.w_ox + h_prev @ model.w_oh + model.b_o)
				c_t = torch.mul(g_t, i_t) + torch.mul(c_prev, f_t)
				h_t = torch.mul(torch.tanh(c_t), o_t)
				hs.append(Variable(h_t, requires_grad=True))
				p_t = h_t @ model.w_ph + model.b_p
				h_prev = h_t
				c_prev = c_t

		if config.model_type == 'RNN':
			h_prev = model.h_t
			for t in range(model.seq_length):
				h_t = torch.tanh(x[:, t, :] @ model.w_hx + h_prev @ model.w_hh + model.b_h)
				hs.append(Variable(h_t, requires_grad=True))
				p_t = h_t @ model.w_ph + model.b_p
				h_prev = h_t

		out = p_t
			
		############################################################################
		torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
		############################################################################

		batch_targets = batch_targets.to(device)
		loss = criterion(out, batch_targets)
		num_correct = (out.argmax(dim=1) == batch_targets).float().cpu()
		accuracy = torch.mean(num_correct, dim=0).item()
		loss.backward(retain_graph=True)
		# No optimizer.step()
		
		# Just for time measurement
		t2 = time.time()
		examples_per_second = config.batch_size/float(t2-t1)

		if step % 10 == 0:

			print("[{}] Exp num {:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
				  "Accuracy = {:.2f}, Loss = {:.3f}, Palindrome length = {}".format(
					datetime.now().strftime("%Y-%m-%d %H:%M"), exp_num,
					config.batch_size, examples_per_second,
					accuracy, loss, config.input_length+1
			))

		# Computing norm grads of hidden states
	
		for i, h in enumerate(hs):
			h = torch.norm(h).cpu().item()
			logger.writerow([exp_num, config.input_length, i, h])
			log_file.flush()

		if step == config.train_steps:
			# If you receive a PyTorch data-loader error, check this bug report:
			# https://github.com/pytorch/pytorch/pull/9655
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
	parser.add_argument('--train_steps', type=int, default=0, help='Number of training steps')
	parser.add_argument('--max_norm', type=float, default=10.0)
	parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

	parser.add_argument('--results_folder', type=str, default='sample_folder_grads_over_time', help="Where to store logs and/or plots")
	parser.add_argument('--log_file', type=str, help="Test log file")
	parser.add_argument('--manual_seed', type=int, default=1, help="Torch manual seed")
	config = parser.parse_args()
 
  
	# obtain the gradients
	conf = config
	conf.model_type = 'RNN'
	for exp_num in range(100):
		config_RNN, results_folder = obtain_grads_over_time(conf, exp_num)
	print("config rnn:", config_RNN.log_file)
	log_file_RNN = config_RNN.log_file

	conf.model_type = 'LSTM'
	conf.log_file = ''
	for exp_num in range(100):
		config_LSTM, results_folder = obtain_grads_over_time(conf, exp_num)
	print("config lstm:", config_LSTM.log_file)

	print('Done obtaining grads over time.')

	plot_grads_timesteps(results_folder, log_file_RNN, config_LSTM.log_file, config.input_length)