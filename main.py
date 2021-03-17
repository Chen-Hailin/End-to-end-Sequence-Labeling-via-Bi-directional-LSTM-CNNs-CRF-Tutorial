from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import autograd

import time
import _pickle as cPickle

import urllib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 80
plt.style.use('seaborn-pastel')
import os
import sys
import codecs
import re
import numpy as np
from config import *
from preprocess import *
from model import *
from utils import *


# read embedding
with open(mapping_file, 'rb') as f:
	mappings = cPickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
char_to_id = mappings['char_to_id']
word_embeds = mappings['word_embeds']


#creating the model using the Class defined above
model = Model_CRF(vocab_size=len(word_to_id),
				   tag_to_ix=tag_to_id,
				   embedding_dim=parameters['word_dim'],
				   hidden_dim=parameters['word_lstm_dim'],
				   use_gpu=use_gpu,
				   char_to_ix=char_to_id,
				   pre_word_embeds=word_embeds,
				   use_crf=parameters['crf'],
				   char_mode=parameters['char_mode'],
				   model_mode=parameters['model_mode'])
print("Model Initialized!!!")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {count_parameters(model)}')

#Reload a saved model, if parameter["reload"] is set to a path
if parameters['reload']:
	if not os.path.exists(parameters['reload']):
		print("downloading pre-trained model")
		model_url="https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/raw/master/trained-model-cpu"
		urllib.request.urlretrieve(model_url, parameters['reload'])
	model.load_state_dict(torch.load(parameters['reload']))
	print("model reloaded :", parameters['reload'])

if use_gpu:
	model.cuda()
#Initializing the optimizer
#The best results in the paper where achived using stochastic gradient descent (SGD) 
#learning rate=0.015 and momentum=0.9 
#decay_rate=0.05 

learning_rate = 0.015
momentum = 0.9
number_of_epochs = parameters['epoch'] 
decay_rate = 0.05
gradient_clip = parameters['gradient_clip']
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

#variables which will used in training process
losses = [] #list to store all losses
loss = 0.0 #Loss Initializatoin
best_dev_F = -1.0 # Current best F-1 Score on Dev Set
best_test_F = -1.0 # Current best F-1 Score on Test Set
best_train_F = -1.0 # Current best F-1 Score on Train Set
all_F = [[0, 0, 0]] # List storing all the F-1 Scores
eval_every = len(train_data) # Calculate F-1 Score after this many iterations
plot_every = 2000 # Store loss after this many iterations
count = 0 #Counts the number of iterations
#parameters['reload']=False

if not parameters['reload']:
	tr = time.time()
	model.train(True)
	for epoch in range(1,number_of_epochs):
		for i, index in enumerate(np.random.permutation(len(train_data))):
			count += 1
			data = train_data[index]

			##gradient updates for each data entry
			model.zero_grad()

			sentence_in = data['words']
			sentence_in = Variable(torch.LongTensor(sentence_in))
			tags = data['tags']
			chars2 = data['chars']
			
			if parameters['char_mode'] == 'LSTM':
				chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
				d = {}
				for i, ci in enumerate(chars2):
					for j, cj in enumerate(chars2_sorted):
						if ci == cj and not j in d and not i in d.values():
							d[j] = i
							continue
				chars2_length = [len(c) for c in chars2_sorted]
				char_maxl = max(chars2_length)
				chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
				for i, c in enumerate(chars2_sorted):
					chars2_mask[i, :chars2_length[i]] = c
				chars2_mask = Variable(torch.LongTensor(chars2_mask))
			
			if parameters['char_mode'] == 'CNN':

				d = {}

				## Padding the each word to max word size of that sentence
				chars2_length = [len(c) for c in chars2]
				char_maxl = max(chars2_length)
				chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
				for i, c in enumerate(chars2):
					chars2_mask[i, :chars2_length[i]] = c
				chars2_mask = Variable(torch.LongTensor(chars2_mask))


			targets = torch.LongTensor(tags)

			#we calculate the negative log-likelihood for the predicted tags using the predefined function
			if use_gpu:
				neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length, d)
			else:
				neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)
			loss += neg_log_likelihood.data.item() / len(data['words'])
			neg_log_likelihood.backward()

			#we use gradient clipping to avoid exploding gradients
			torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
			optimizer.step()

			#Storing loss
			if count % plot_every == 0:
				loss /= plot_every
				print(count, ': ', loss)
				if losses == []:
					losses.append(loss)
				losses.append(loss)
				loss = 0.0

			#Evaluating on Train, Test, Dev Sets
			if count % (eval_every) == 0 and count > (eval_every * 20) or \
					count % (eval_every*4) == 0 and count < (eval_every * 20):
				model.train(False)
				best_train_F, new_train_F, _ = evaluating(model, train_data, best_train_F, tag_to_id, "Train")
				best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F, tag_to_id, "Dev")
				if save:
					print("Saving Model to ", model_name)
					torch.save(model.state_dict(), model_name)
				best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F, tag_to_id, "Test")

				all_F.append([new_train_F, new_dev_F, new_test_F])
				model.train(True)

			#Performing decay on the learning rate
			if count % len(train_data) == 0:
				adjust_learning_rate(optimizer, lr=learning_rate/(1+decay_rate*count/len(train_data)))

	print(time.time() - tr)
	plt.plot(losses)
	plt.show()
	plt.savefig(f"{parameters['name']}.png")

if not parameters['reload']:
	#reload the best model saved from training
	model.load_state_dict(torch.load(model_name))