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

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

	def __init__(self, batch_size, seq_length, vocabulary_size, dropout_keep_prob,
				 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

		super(TextGenerationModel, self).__init__()
		self.batch_size = batch_size
		self.seq_length = seq_length # 30
		self.lstm_num_hidden = lstm_num_hidden # 128
		self.vocabulary_size = vocabulary_size # num_classes so 87
		self.device = device
		self.lstm_num_layers = lstm_num_layers
		self.layers = nn.ModuleList()
		
		self.lstm = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers, batch_first=True, dropout=1-dropout_keep_prob) #
		self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

		self.h_0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_num_hidden).to(device)
		self.c_0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.lstm_num_hidden).to(device)

	def forward(self, x):
		h_t = self.h_0
		c_t = self.c_0
		out, (h_t, c_t) = self.lstm(x, (h_t, c_t)) 
		output = self.linear(out)  
		return output
