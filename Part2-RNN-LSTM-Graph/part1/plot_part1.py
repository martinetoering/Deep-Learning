import csv
import numpy as np
import matplotlib.pyplot as plt
import os

import argparse
import pandas as pd

def create_df(results_folder, file_name):
	file_path = os.path.join(results_folder, file_name)
	df = pd.read_csv(file_path, sep='\t')
	df = df.to_numpy()
	return df

def plot(results_folder, plot_path, title, x_values, y_values, x_labels="Steps", y_labels="Accuracy",
		ylim=None, xlim=None, xticks=None):
	plot_path = os.path.join(results_folder, plot_path)
	plt.close("all")
	plt.figure()
	plt.plot(x_values, y_values)
	plt.title(title, fontsize=13)
	plt.xlabel(x_labels, fontsize=14)
	plt.ylabel(y_labels, fontsize=14)
	plt.grid(True)
	plt.ylim(ylim)
	plt.xlim(xlim)
	plt.xticks(xticks)
	plt.savefig(plot_path)

def plot_train_part1(model_type, results_folder, file_name, input_length):
	print("Plot train")
	file_name = os.path.join(results_folder, file_name)
	steps = []
	loss = []
	acc = []
	with open(file_name, newline='') as file:
		file_reader = csv.reader(file, delimiter='\t')
		next(file_reader, None) # skip header
		for item in file_reader:
			steps.append(int(item[0]))
			loss.append(float(item[1]))
			acc.append(float(item[2]))
		print("Log file read")

	max_step = steps[-1]
	print("Last step:", max_step)
	xlim = (0-(0.01*max_step), max_step+(max_step*0.01))
	ylim = None 

	loss_plot_path = "train_loss_{}_palindrome_{}.png".format(model_type, input_length+1)
	loss_title = "{} Loss\n Palindrome length = {}".format(model_type, input_length+1)
	plot(results_folder, loss_plot_path, loss_title, steps, loss, ylim=ylim, xlim=xlim)

	ylim = (-0.05, 1.05)

	acc_plot_path = "train_acc_{}_palindrome_{}.png".format(model_type, input_length+1)
	acc_title = "{} Accuracy\n Palindrome length = {}".format(model_type, input_length+1)
	plot(results_folder, acc_plot_path, acc_title, steps, acc, ylim=ylim, xlim=xlim)


	print("Done plot train")

def plot_test_part1(model_type, results_folder, file_name):
	print("Plot test")
	# input_lengths = []
	# acc = []
	log = create_df(results_folder, file_name)
	# Sort unique input lengths
	input_lengths = np.sort(np.unique(log[:, 0])).astype(int)
	acc = np.zeros(input_lengths.shape[0])
	for i, item in enumerate(input_lengths):
		# Take mean over experiments
		acc[i] = np.mean(log[np.where(log[:, 0] == item), 2])

	max_input_length = input_lengths[-1]
	print("Last input length:", max_input_length)

	ylim = (-0.05, 1.05)
	xlim = (2-(0.01*max_input_length), max_input_length+(max_input_length*0.01))
	xticks = list(range(2, max_input_length+2, 4))
	plot_path = "test_{}.png".format(model_type)
	title = "{} Accuracy vs Palindrome length".format(model_type)

	plot(results_folder, plot_path, title, input_lengths, acc, x_labels="Palindrome length",
		ylim=ylim, xlim=xlim, xticks=xticks)

	print("Done plot test")

def plot_grads_timesteps(results_folder, file_name0, file_name1, input_length):
	print("Plot gradient norms between timesteps")
	log = create_df(results_folder, file_name0)
	input_lengths = np.sort(np.unique(log[:, 2])).astype(int)
	print("timesteps:", input_lengths)
	norm0 = np.zeros(input_lengths.shape[0])
	for i, item in enumerate(input_lengths):
		norm0[i] = np.mean(log[np.where(log[:, 2] == item), 3])

	log = create_df(results_folder, file_name1)
	input_lengths = np.sort(np.unique(log[:, 2])).astype(int)
	print("timesteps:", input_lengths)
	norm1 = np.zeros(input_lengths.shape[0])
	for i, item in enumerate(input_lengths):
		norm1[i] = np.mean(log[np.where(log[:, 2] == item), 3])

	plt.close("all")
	print("norm0:", norm0)
	print("norm1:", norm1)
	max_input_length = input_lengths[-1]
	print("Last timestep:", max_input_length)
	plt.figure()
	plt.plot(norm0, label="RNN")
	plt.plot(norm1, label="LSTM")
	plt.title("Grads over time", fontsize=13)
	plt.xlabel("Timestep", fontsize=14)
	plt.ylabel("Norm of gradient of h", fontsize=14)
	plt.grid(True)
	plt.legend()
	result_path = os.path.join(results_folder, "grads_over_time_{}.png".format(input_length))
	plt.savefig(result_path)
	print("Saved at:", results_folder)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--results_folder', type=str, default='results', 
    	help="Where logs are stored and plots need to be stored")
    parser.add_argument('--log_file', type=str, help="Train log file")
    parser.add_argument('--input_length', type=int, default=4, help="Input length")
    parser.add_argument('--test_log_file', type=str, help="Test log file")

    config = parser.parse_args()

    if config.log_file:
    	plot_train_part1(config.model_type, config.results_folder, config.log_file, config.input_length)
    if config.test_log_file:
    	plot_test_part1(config.model_type, config.results_folder, config.test_log_file)
