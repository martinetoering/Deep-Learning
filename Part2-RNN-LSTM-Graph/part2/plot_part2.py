import csv
import numpy as np
import matplotlib.pyplot as plt
import os

import argparse
import pandas as pd


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
	plt.savefig(plot_path)

def plot_train_part2(model_type, results_folder, file_name, input_length):
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
	ylim = None
	xlim = (0-(0.01*max_step), max_step+(max_step*0.01))

	loss_plot_path = "train_loss_{}_sentence_{}.png".format(model_type, input_length)
	loss_title = "LSTM ({}) Loss\n Sentence length = {}".format(model_type, input_length)
	plot(results_folder, loss_plot_path, loss_title, steps, loss, ylim=ylim, xlim=xlim)

	ylim = (-0.05, 1.05)

	acc_plot_path = "train_acc_{}_sentence_{}.png".format(model_type, input_length)
	acc_title = "LSTM ({}) Accuracy\n Sentence length = {}".format(model_type, input_length)
	plot(results_folder, acc_plot_path, acc_title, steps, acc, ylim=ylim, xlim=xlim)

	print("Done plot train")


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--results_folder', type=str, default='results', 
    	help="Where logs are stored and plots need to be stored")
    parser.add_argument('--log_file', type=str, help="Train log file")
    parser.add_argument('--input_length', type=int, default=30, help="Input length")
    parser.add_argument('--test_log_file', type=str, help="Test log file")

  
    config = parser.parse_args()

    # Train the model
    if config.log_file:
    	plot_train_part2(config.model_type, config.results_folder, config.log_file, config.input_length)
    if config.test_log_file:
    	plot_test_part2(config.model_type, config.results_folder, config.test_log_file)
