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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

	def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
		super(VanillaRNN, self).__init__()
		self.seq_length = seq_length 
		self.input_dim = input_dim
		self.num_hidden = num_hidden
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.device = device

		# input-to-hidden 
		self.w_hx = torch.nn.Parameter(torch.Tensor(input_dim, num_hidden).to(device))
		# hidden-to-hidden
		self.w_hh = torch.nn.Parameter(torch.Tensor(num_hidden, num_hidden).to(device))
		# hidden-to-output
		self.w_ph = torch.nn.Parameter(torch.Tensor(num_hidden, num_classes).to(device))
		self.b_h = torch.nn.Parameter(torch.Tensor(num_hidden).to(device))
		self.b_p = torch.nn.Parameter(torch.Tensor(num_classes).to(device))

		# Initialization 
		self.w_hx = nn.init.kaiming_normal_(self.w_hx)
		self.w_hh = nn.init.kaiming_normal_(self.w_hh)
		self.w_ph = nn.init.kaiming_normal_(self.w_ph)
		self.b_h = nn.init.constant_(self.b_h, 0)
		self.b_p = nn.init.constant_(self.b_p, 0)

		# Initialize h(0)
		# h of batch size (default 128) times num_hidden (default 128)
		self.h_t = torch.zeros((self.batch_size, self.num_hidden)).to(device)

	def forward(self, x):
		h_prev = self.h_t
		for t in range(self.seq_length):
			# h consists for default of tanh of 128x10 (x) times 10x128 (w_hx) + 128x128 times 128x128
			h_t = torch.tanh(x[:, t, :] @ self.w_hx + h_prev @ self.w_hh + self.b_h)
			# for default 128x128 (h_t) times 128x10 (w_ph)
			p_t = h_t @ self.w_ph + self.b_p
			h_prev = h_t
		return p_t
