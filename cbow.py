import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchtext
import torchtext.legacy.data as tld

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
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


def preprocess(x):
	
	""" Function to remove punctuation and unwanted symbols from text """

	res = re.sub(r'[^\w\s]', '', x)
	res = res.strip('\n')

	return res


def train(model, optimizer, train_iterator, test_iterator=None, loss=nn.NLLLoss(), epochs=20):

	"""
	Function to train our CBOW model

	Parameters:
	-----------
	model : model.CBOW
		Represents the CBOW model

	optimizer : torch.optim
		Represents the optimizer for the model optimization

	train_iterator : torchtext.legacy.data.iterator.Iterator
		Represents the batch iterator for training the data

	test_iterator : torchtext.legacy.data.iterator.Iterator, optional; default:None
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

	train_losses = list()
	test_losses = list()

	for epoch in range(epochs):

		tl = list()
		el = list()

		""" Training the model """
		model.train()
		for c, t in tqdm(train_iterator):
			c = c.to(device)
			t = t.to(device)

			optimizer.zero_grad()
			pred = model(c)
			_l = loss(pred, t)
			_l.backward()
			optimizer.step()

			tl.append(_l.item())

		if test_iterator is not None:
			""" Evaluating the model """
			model.eval()
			for c, t in tqdm(test_iterator):
				
				with torch.no_grad():
					c = c.to(device)
					t = t.to(device)

					pred = model(c)
					_l = loss(pred, t)
					el.append(_l.item())


		tl = np.mean(tl)
		train_losses.append(tl)
		if test_iterator is not None:
			el = np.mean(el)
			test_losses.append(el)

			print(f'Epoch : {epoch+1} / {epochs}, Train Loss : {tl}, Test Loss : {el}')

		else:
			print(f'Epoch : {epoch+1} / {epochs}, Train Loss : {tl}')


	return train_losses, test_losses




if __name__ == "__main__":

	""" Reading the data """
	with open('./DATA/snow_white.txt', 'r') as f:
		raw_text = f.readlines()

	""" Preprocessing the text data """
	ptext1 = list()
	for i in tqdm(range(len(raw_text))):
		ptext1.append(preprocess(raw_text[i]))

	ptext1 = list(filter(lambda x: x != '', ptext1))
	
	final_text = list()
	for t in ptext1:
		t = t.split()
		for w in t:
			final_text.append(w)

	
	cntx2tgt = build_context_target(final_text)

	df_train, df_test = train_test_split(cntx2tgt, test_size=.02, shuffle=True, random_state=42)
	df_train.to_csv('./DATA/train_data.csv', index=False)
	df_test.to_csv('./DATA/test_data.csv', index=False)

	TEXT = tld.Field(sequential=True, batch_first=True, pad_first=True, tokenize=str.split, lower=True, use_vocab=True)
	LABEL = tld.Field(sequential=False, is_target=True, use_vocab=True)

	train_datasets = tld.TabularDataset(path='./DATA/train_data.csv', format='csv',
		skip_header=True, fields=[('context', TEXT), ('targets', LABEL)])

	
	test_datasets = tld.TabularDataset(path='./DATA/test_data.csv', format='csv',
		skip_header=True, fields=[('context', TEXT), ('targets', LABEL)])

	TEXT.build_vocab(train_datasets, min_freq=5)
	LABEL.build_vocab(train_datasets, min_freq=5)
	vocab = TEXT.vocab

	print(f'Length of the vocabulary : {len(vocab)}')

	train_batch_iter = tld.Iterator(train_datasets, batch_size=64, sort_key=lambda x: len(x.context), train=True, shuffle=True)
	test_batch_iter = tld.Iterator(test_datasets, batch_size=64, sort_key=lambda x: len(x.context), train=False, shuffle=False)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	print(f'Device used : {device}')
	
	model = CBOW(len(vocab), 100).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
	train_losses, test_losses = train(model, optimizer, train_batch_iter, test_iterator=test_batch_iter, epochs=100)



	plt.figure(figsize=(12, 8))
	plt.plot(train_losses, label='Train Loss')
	plt.plot(test_losses, label='Test Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Batched-Losses over Epochs')
	plt.legend()
	plt.show()

	print('\n'*10)
	input_data = input('Enter 4 context words: ').lower().split()
	context_vector = np.array([vocab.stoi[i] for i in input_data]).reshape(1, -1)
	ix2word = {v:k for k, v in vocab.stoi.items()}
	prediction = model(torch.from_numpy(context_vector).to(device))[0]
	prediction = ix2word[torch.argmax(prediction).item()]

	print(f'Prediction : {prediction}\n\n')






	


