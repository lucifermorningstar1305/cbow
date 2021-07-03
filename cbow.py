import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchtext
import torchtext.legacy.data as tld

from tqdm import tqdm

from model import CBOW

plt.style.use('ggplot')


def build_context_target(data):

	""" 
	Function to build the Context to Target data
	
	Parameters:
	-----------
	data : array-like
		Represents the string data in list format

	Returns:
	--------
	c2t : pandas.DataFrame
		Represents the mapping of context words to target word
	
	"""

	contexts = list()
	targets = list()

	for i in range(2, len(data) - 2):

		cntx = ' '.join([data[i-2], data[i-1], data[i+1], data[i+2]])
		tgt = data[i]

		contexts.append(cntx)
		targets.append(tgt)

	c2t = pd.DataFrame({'context' : contexts, 'targets':targets})

	return c2t


def train(model, optimizer, batch_iterator, loss=nn.NLLLoss(), epochs=20):

	"""
	Function to train our CBOW model

	Parameters:
	-----------
	model : model.CBOW
		Represents the CBOW model

	optimizer : torch.optim
		Represents the optimizer for the model optimization

	batch_iter : torchtext.legacy.data.iterator.Iterator
		Represents the batch iterator for training the data

	loss : torch.nn.modules.loss, optional; default:nn.NLLLoss()
		Represents the loss function to use for training

	epochs : int, optional; default:20
		Represents the number of training epochs.

	Returns:
	-------
	losses : list
		Represents the losses encountered in each batch
	"""

	losses = list()

	for epoch in range(epochs):
		l = list()

		for c, t in tqdm(batch_iterator):

			c = c.to(device)
			t = t.to(device)
		
			optimizer.zero_grad()

			pred = model(c)
			_l = loss(pred, t)
			_l.backward()
			optimizer.step()

			l.append(_l.item())

		losses.append(np.mean(l))

		print(f'Epoch : {epoch+1} / {epochs}, Loss : {np.mean(l)}')


	return losses




if __name__ == "__main__":

	raw_text = """We are about to study the idea of a computational process.
	Computational processes are abstract beings that inhabit computers.
	As they evolve, processes manipulate other abstract things called data.
	The evolution of a process is directed by a pattern of rules
	called a program. People create programs to direct processes. In effect,
	we conjure the spirits of the computer with our spells.""".split()

	cntx2tgt = build_context_target(raw_text)
	cntx2tgt.to_csv('./context2target.csv', index=False)

	TEXT = tld.Field(sequential=True, batch_first=True, pad_first=True, tokenize=str.split, lower=True, use_vocab=True)
	LABEL = tld.Field(sequential=False, is_target=True, use_vocab=True)

	datasets = tld.TabularDataset(path='./context2target.csv', format='csv',
		skip_header=True, fields=[('context', TEXT), ('targets', LABEL)])

	TEXT.build_vocab(datasets, min_freq=2)
	LABEL.build_vocab(datasets)
	vocab = TEXT.vocab

	print(f'Length of the vocabulary : {len(vocab)}')

	batch_iter = tld.Iterator(datasets, batch_size=64, sort_key=lambda x: len(x.context), train=True, shuffle=True)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	print(f'Device used : {device}')
	
	model = CBOW(len(vocab), 100).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
	losses = train(model, optimizer, batch_iter, epochs=100)



	# plt.figure(figsize=(12, 8))
	# plt.plot(losses)
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.title('Batched-Losses over Epochs')
	# plt.show()

	print('\n'*10)
	input_data = input('Enter 4 context words: ').lower().split()
	context_vector = np.array([vocab.stoi[i] for i in input_data]).reshape(1, -1)
	ix2word = {v:k for k, v in vocab.stoi.items()}
	prediction = model(torch.from_numpy(context_vector).to(device))[0]
	prediction = ix2word[torch.argmax(prediction).item()]

	print(f'Prediction : {prediction}\n\n')



	


