from collections import OrderedDict
import torch
#parameters for the Model
parameters = OrderedDict()
parameters['train'] = "./data/eng.train" #Path to train file
parameters['dev'] = "./data/eng.testa" #Path to test file
parameters['test'] = "./data/eng.testb" #Path to dev file
parameters['tag_scheme'] = "BIOES" #BIO or BIOES
parameters['lower'] = True # Boolean variable to control lowercasing of words
parameters['zeros'] =  True # Boolean variable to control replacement of  all digits by 0 
parameters['char_dim'] = 30 #Char embedding dimension
parameters['word_dim'] = 100 #Token embedding dimension
parameters['word_lstm_dim'] = 200 #Token LSTM hidden layer size
parameters['word_bidirect'] = True #Use a bidirectional LSTM for words
parameters['embedding_path'] = "./data/glove.6B.100d.txt" #Location of pretrained embeddings
parameters['all_emb'] = 1 #Load all embeddings
parameters['crf'] =1 #Use CRF (0 to disable)
parameters['dropout'] = 0.5 #Droupout on the input (0 = no dropout)
parameters['epoch'] =  50 #Number of epochs to run"
parameters['weights'] = "" #path to Pretrained for from a previous run
parameters['name'] = "self-trained-relu-CNN_CNN1" # Model name
parameters['gradient_clip']=5.0
parameters['char_mode']="CNN"
parameters['model_mode']="CNN" #  [LSTM, CNN]
parameters['CNN_params'] = {'kernel_size':[3,3,3], 'dilation': [1,2,3], 'n_layers':1}
models_path = "./models/" #path to saved models

#GPU
parameters['use_gpu'] = torch.cuda.is_available() #GPU Check
use_gpu = parameters['use_gpu']

parameters['reload'] = ''#"./models/pre-trained-model" 

#Constants
START_TAG = '<START>'
STOP_TAG = '<STOP>'
#paths to files 
#To stored mapping file
mapping_file = './data/mapping.pkl'

#To stored model
name = parameters['name']
model_name = models_path + name #get_name(parameters)