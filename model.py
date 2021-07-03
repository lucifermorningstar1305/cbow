import numpy as np
import torch
import torch.nn as nn

class CBOW(nn.Module):

	"""
	A Wrapper class to implement the CBOW method.

	Parameters:
	-----------
	vocab_size : int
		Represents the vocabulary size of the data

	embedd_dim : int
		Represents the embedding dimension of the Embedding matrix

	hidden_dim : int, optional; default:128
		Represents the hidden dimension.

	
	Returns:
	--------
	res : str
		Represents the central word of the data.

	References:
	-----------
	 [1] https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py
	"""

	def __init__(self, vocab_size, embedd_dim, hidden_dim=128):

		super(CBOW, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedd_dim)
		self.linear = nn.Linear(embedd_dim, hidden_dim)
		self.activation1 = nn.ReLU()

		self.out = nn.Linear(hidden_dim, vocab_size)
		self.activation2 = nn.LogSoftmax(dim=-1)


	def forward(self, X):
		embedd = torch.sum(self.embedding(X), dim=1)
		hidden_res = self.linear(embedd)
		hidden_res = self.activation1(hidden_res)

		res = self.out(hidden_res)
		res = self.activation2(res)

		return res

