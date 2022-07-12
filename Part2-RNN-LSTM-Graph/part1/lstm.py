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

class LSTM(nn.Module):

	def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
		super(LSTM, self).__init__()
		self.seq_length = seq_length
		self.input_dim = input_dim
		self.num_hidden = num_hidden
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.device = device

		self.w_gx = torch.nn.Parameter(torch.Tensor(input_dim, num_hidden).to(device))
		self.w_gh = torch.nn.Parameter(torch.Tensor(num_hidden, num_hidden).to(device))
		self.b_g = torch.nn.Parameter(torch.Tensor(num_hidden).to(device))

		self.w_ix = torch.nn.Parameter(torch.Tensor(input_dim, num_hidden).to(device))
		self.w_ih = torch.nn.Parameter(torch.Tensor(num_hidden, num_hidden).to(device))
		self.b_i = torch.nn.Parameter(torch.Tensor(num_hidden).to(device))

		self.w_fx = torch.nn.Parameter(torch.Tensor(input_dim, num_hidden).to(device))
		self.w_fh = torch.nn.Parameter(torch.Tensor(num_hidden, num_hidden).to(device))
		self.b_f = torch.nn.Parameter(torch.Tensor(num_hidden).to(device))

		self.w_ox = torch.nn.Parameter(torch.Tensor(input_dim, num_hidden).to(device))
		self.w_oh = torch.nn.Parameter(torch.Tensor(num_hidden, num_hidden).to(device))
		self.b_o = torch.nn.Parameter(torch.Tensor(num_hidden).to(device))

		self.w_ph = torch.nn.Parameter(torch.Tensor(num_hidden, num_classes).to(device))
		self.b_p = torch.nn.Parameter(torch.Tensor(num_classes).to(device))

		# Initialization 
		self.w_gx = nn.init.kaiming_normal_(self.w_gx)
		self.w_gh = nn.init.kaiming_normal_(self.w_gh)
		self.b_g = nn.init.constant_(self.b_g, 0)
		self.w_ix = nn.init.kaiming_normal_(self.w_ix)
		self.w_ih = nn.init.kaiming_normal_(self.w_ih)
		self.b_i = nn.init.constant_(self.b_i, 0)
		self.w_fx = nn.init.kaiming_normal_(self.w_fx)
		self.w_fh = nn.init.kaiming_normal_(self.w_fh)
		self.b_f = nn.init.constant_(self.b_f, 0)
		self.w_ox = nn.init.kaiming_normal_(self.w_ox)
		self.w_oh = nn.init.kaiming_normal_(self.w_oh)
		self.b_o = nn.init.constant_(self.b_o, 0)
		self.w_ph = nn.init.kaiming_normal_(self.w_ph)
		self.b_p = nn.init.constant_(self.b_p, 0)

		# Initialize h(0)
		self.h_t = torch.zeros((self.batch_size, self.num_hidden)).to(device)
		# Initialize c(0)
		self.c_t = torch.zeros((self.batch_size, self.num_hidden)).to(device)


	def forward(self, x):
		h_prev = self.h_t
		c_prev = self.c_t
		for t in range(self.seq_length):
			g_t = torch.tanh(x[:, t, :] @ self.w_gx + h_prev @ self.w_gh + self.b_g)
			i_t = torch.sigmoid(x[:, t, :] @ self.w_ix + h_prev @ self.w_ih + self.b_i)
			f_t = torch.sigmoid(x[:, t, :] @ self.w_fx + h_prev @ self.w_fh + self.b_f)
			o_t = torch.sigmoid(x[:, t, :] @ self.w_ox + h_prev @ self.w_oh + self.b_o)
			c_t = torch.mul(g_t, i_t) + torch.mul(c_prev, f_t)
			h_t = torch.mul(torch.tanh(c_t), o_t)
			p_t = h_t @ self.w_ph + self.b_p
			h_prev = h_t
			c_prev = c_t
		return p_t