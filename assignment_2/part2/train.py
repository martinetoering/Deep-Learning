# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

from part2.plot_part2 import plot_train_part2

import csv
from torch.optim import lr_scheduler

################################################################################

def save_model(results_folder, step, train_loss, train_accuracy, seq_length,
					model_state_dict, optimizer_state_dict):
	save_file = os.path.join(results_folder, 'save_{}.pth'.format(step))
	states = {
		'step': step + 1,
		'train_loss': train_loss,
		'train_accuracy': train_accuracy,
		'sentence_length': seq_length,
		'state_dict': model_state_dict,
		'optimizer': optimizer_state_dict,
	}
	torch.save(states, save_file)

def train(config):

	config.train_steps = int(config.train_steps)

	# Initialize the device which to run the model on
	device = torch.device(config.device)
	print("device:", device)

	# Initialize the dataset and data loader
	dataset = TextDataset(config.txt_file, config.seq_length)
	data_loader = DataLoader(dataset, config.batch_size, num_workers=1, shuffle=True, drop_last=True)
	vocab_size = dataset.vocab_size

	print("Sample sentences length:", config.sample_sentences_length)
	test_dataset = TextDataset(config.txt_file, config.sample_sentences_length)
	test_data_loader = DataLoader(dataset, config.batch_size, num_workers=1, shuffle=True, drop_last=True)

	finish_dataset = TextDataset(config.txt_file, config.finish_sentences_length*2)
	finish_data_loader = DataLoader(dataset, config.batch_size, num_workers=1, shuffle=True, drop_last=True)

	# Initialize the model that we are going to use
	model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.dropout_keep_prob, 
		config.lstm_num_hidden, config.lstm_num_layers, device)
	model.to(device)

	torch.manual_seed(config.manual_seed)

	# Results and logging
	results_folder = os.path.join(os.getcwd(), '{}_{}'.format(config.results_folder, config.seq_length))
	if not os.path.exists(results_folder):
		os.makedirs(results_folder)
		print("Created results dir: {}".format(results_folder))
	file_name = "sentence_{}".format(config.seq_length)
	if not config.no_train:
		config.log_file = os.path.join(results_folder, "train_{}.log".format(file_name))
		log_file = open(config.log_file, "w+")
		logger = csv.writer(log_file, delimiter='\t')
		logger.writerow(["step", "loss", "accuracy"])
		if config.generate_sentences:
			g_sentence_file = open(os.path.join(results_folder, "generated_sentences_{}.txt".format(file_name)), "w+")
			g_sentence_logger = csv.writer(g_sentence_file, delimiter='\t')
			g_sentence_logger.writerow(["step", "num", "sentence"])
	if config.finish_sentences:
		f_sentence_file = open(os.path.join(results_folder, "finished_sentences_{}.txt".format(file_name)), "w+")
		f_sentence_logger = csv.writer(f_sentence_file, delimiter='\t')
		f_sentence_logger.writerow(["num", "input", "sentence"])

	# Store config 
	config_file_name = os.path.join(results_folder, "config_{}.txt".format(config.seq_length))
	if not os.path.exists(config_file_name):
		with open(config_file_name, 'w') as config_file:
			print(vars(config), file=config_file)

	# Setup the loss and optimizer
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

	if config.resume_path:
		# Resume from saved model
		resume_path = os.path.join(results_folder, config.resume_path)
		print("Loading saved model {}".format(resume_path))
		checkpoint = torch.load(resume_path)
		step = checkpoint['step']
		train_loss = checkpoint['train_loss']
		train_accuracy = checkpoint['train_accuracy']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
	else:
		step = 0
	
	train_loss = []
	train_accuracy = []

	finish_data_l = iter(finish_data_loader)

	if not config.no_train:
		# Some modifications because of dataloader stopping at step (#chars/batch size)
		convergence = False
		n_epochs = int(config.train_steps/len(data_loader))+1
		print("n_epochs:", n_epochs)
		for i in range(n_epochs):
			data_l = iter(data_loader)
			test_data_l = iter(test_data_loader)
			for (batch_inputs, batch_targets) in data_l:

				# Only for time measurement of step through network
				t1 = time.time()

				model.train()

				batch_i = batch_inputs
				optimizer.zero_grad()
				# Batch inputs to one-hot
				onehot_batch_inputs = torch.eye(vocab_size)[batch_inputs.long()]
				batch_inputs = onehot_batch_inputs.to(device)
				out = model(batch_inputs)
				# Seq length to middle for crossentropy over timesteps
				out = out.permute(0, 2, 1)

				batch_targets = batch_targets.to(device)
				 # Reduction on output is mean on default
				loss = criterion(out, batch_targets)
				num_correct = (out.argmax(dim=1) == batch_targets).float().cpu()
				accuracy_timesteps = torch.mean(num_correct, dim=0)
				accuracy = torch.mean(accuracy_timesteps, dim=0).item()
				loss.backward()
				optimizer.step()
				train_loss.append(loss.cpu().item())
				train_accuracy.append(accuracy)

				# Just for time measurement
				t2 = time.time()
				examples_per_second = config.batch_size/float(t2-t1)

				if step % config.print_every == 0:
					
					print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
						  "Accuracy = {:.2f}, Loss = {:.3f}, Seq length = {}".format(
							datetime.now().strftime("%Y-%m-%d %H:%M"), step,
							config.train_steps, config.batch_size, examples_per_second,
							accuracy, loss, config.seq_length
					))

					logger.writerow([step, loss.cpu().item(), accuracy])
					log_file.flush()
					# If train convergence set to true we allow early stopping if loss does not decrease
					if config.train_convergence:
						if step > config.steps_convergence_train:
							if min(train_loss) < min(train_loss[step-config.steps_convergence_train:step]):
								print("Convergence at {}, stopping train".format(step))
								convergence = True

				if step % config.sample_every == 0:
					
					model = model.eval()
					print("Generating some sentences by sampling from the model...")
					# Generate some sentences by sampling from the model
					l = config.sample_sentences_length
					(test_inputs, test_inputs) = test_data_l.next()
					onehot_test_inputs = torch.eye(vocab_size)[test_inputs.long()]
					onehot_test_inputs = onehot_test_inputs.to(device)
					test_sentences = test_inputs.numpy()
					# Store output sentences for each batch (row)
					out_sentences = np.zeros((config.batch_size, config.seq_length))
					out_tensor = torch.zeros((config.batch_size, l, vocab_size)).to(device)
					out_tensor[:, 0, :] = onehot_test_inputs[:, 0, :]
					k = 0
					for t in range(l-1):
						output = model(onehot_test_inputs)
						if t > (config.seq_length-1):
							output = output[:, config.seq_length-1, :]
						else:
							output = output[:, t, :]
						# Greedy sampling (temperature = 1) or random sampling
						probs = torch.softmax(output*config.temperature, dim=1)
						dis = torch.distributions.categorical.Categorical(probs=probs)
						if config.greedy_sampling:
							pred_chars = torch.argmax(probs, dim=1)
						else:
							pred_chars = dis.sample()
						out_tensor[:, t+1, :] = torch.eye(vocab_size)[pred_chars.long()]
						if t > (config.seq_length-2):
							k += 1
							onehot_test_inputs = out_tensor[:, k:k+config.seq_length, :]
						else:
							onehot_test_inputs[:, t+1, :] = out_tensor[:, t+1, :]	
					test_outputs = torch.argmax(out_tensor, dim=2)
					out_sentences = test_outputs.cpu().numpy()
					# Loop over batches
					for batch_num, sentence in enumerate(out_sentences):
						out_sentence = test_dataset.convert_to_string(sentence)
						# Store in generated sentence log the train step
						g_sentence_logger.writerow([step, batch_num, out_sentence])
						g_sentence_file.flush()
						# Print some
						if batch_num < 5:
							in_sentence = test_dataset.convert_to_string(test_sentences[batch_num, :])						
							# print("Input {}: {}".format(batch_num, in_sentence))
							print("Sentence {}: {}".format(batch_num, out_sentence))

				if step % config.checkpoint == 0:
					save_model(results_folder, step, train_loss, train_accuracy, config.seq_length,
						model.state_dict(), optimizer.state_dict()) 

				step += 1
				scheduler.step()

			if convergence:
				print("Convergence")
				# If you receive a PyTorch data-loader error, check this bug report:
				# https://github.com/pytorch/pytorch/pull/9655
				save_model(results_folder, step, train_loss, train_accuracy, config.seq_length,
					model.state_dict(), optimizer.state_dict())
				break

		print('Done training.')

	if config.finish_sentences:
		print('Finishing sentences from the book...')
		# Finish sentences from the book

		model = model.eval()

		for test_step, (finish_inputs, finish_outputs) in enumerate(finish_data_l):
			
			l = config.finish_sentences_length
			onehot_finish_inputs = torch.eye(vocab_size)[finish_inputs.long()]
			onehot_finish_inputs = onehot_finish_inputs.to(device)
			in_sentences = finish_inputs[:, :l].numpy()
			# Store sentences for each batch (row)
			out_sentences = np.zeros((config.batch_size, config.seq_length))
			for t in range(l, config.seq_length-1):
				output = model(onehot_finish_inputs)
				output = output[:, t, :]
				# Greedy sampling (temperature = 1) or random sampling
				probs = torch.softmax(output*config.temperature, dim=1)
				pred_chars = torch.argmax(probs, dim=1)
				onehot_finish_inputs[:, t+1, :] = torch.eye(vocab_size)[pred_chars.long()]
			onehot_finish_inputs = onehot_finish_inputs[:, l:, :]
			finish_outputs = torch.argmax(onehot_finish_inputs, dim=2)
			out_sentences = finish_outputs.cpu().numpy()
			# Loop over batches
			for batch_num, (sentence_i, sentence_o) in enumerate(zip(in_sentences, out_sentences)):
				in_sentence = test_dataset.convert_to_string(sentence_i)
				out_sentence = test_dataset.convert_to_string(sentence_o)
				f_sentence_logger.writerow([step, in_sentence, out_sentence])
				f_sentence_file.flush()
				if batch_num < 5:											
					print("Input {}: {}".format(batch_num, in_sentence))
					print("Sentence {}: {}".format(batch_num, out_sentence))

			if test_step == 10:
				print("Done")
				break

	return config, results_folder


 ################################################################################
 ################################################################################

if __name__ == "__main__":

	# Parse training configuration
	parser = argparse.ArgumentParser()

	# Model params
	parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
	parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
	parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
	parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
	parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

	# Training params
	parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
	parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

	# It is not necessary to implement the following three params, but it may help training.
	parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
	parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
	parser.add_argument('--dropout_keep_prob', type=float, default=0.8, help='Dropout keep probability')

	parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
	parser.add_argument('--max_norm', type=float, default=5.0, help='--')

	# Misc params
	parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
	parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
	parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

	parser.add_argument('--model_type', type=str, default="LSTM")
	parser.add_argument('--results_folder', type=str, default='sample_folder', help="Where to store logs and/or plots")
	parser.add_argument('--log_file', type=str, help="Train log file")
	parser.add_argument('--manual_seed', type=int, default=1, help="Torch manual seed")
	parser.add_argument('--train_convergence', type=bool, default=True, help="Early stopping is possible in train")
	parser.add_argument('--steps_convergence_train', type=int, default=10000,
		help="How many train steps the loss should not decrease for convergence")
	parser.add_argument('--generate_sentences', type=bool, default=True, 
		help="Generate sentences by passing random character")
	parser.add_argument('--sample_sentences_length', type=int, default=30, help="Generate sentences of this length")
	parser.add_argument('--finish_sentences', type=bool, default=True, 
		help="Generate sentences by finishing existing sentences")
	parser.add_argument('--finish_sentences_length', type=int, default=15, help="Finish sentences of this length")
	parser.add_argument('--resume_path', type=str, help="Where to load stored model from (relative path)")
	parser.add_argument('--checkpoint', type=int, default=1000, help="At how many steps store model")
	parser.add_argument('--temperature', type=float, default=1.0, help="Temperature in the softmax")
	parser.add_argument('--no_train', type=bool, default=False, help="If true no training is performed")
	parser.add_argument('--greedy_sampling', type=bool, default=True, 
		help="If true greedy sampling is performed and not random sampling with temperature")


	config = parser.parse_args()

	# Train the model
	config, results_folder = train(config)
	# Uncomment to plot
	# plot_train_part2(config.model_type, results_folder, config.log_file, config.seq_length)