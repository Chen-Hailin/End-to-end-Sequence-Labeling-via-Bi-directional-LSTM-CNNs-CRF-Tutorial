# DeepNLP Assignment 2

This repo is based on [**IPython Notebook of the tutorial**](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb). I make it modular and implement for CNN model with user specified number of layers, support dilated convolution and crf/softmax output layer. 


### Authors

[**Chen Hailin**]

### Installation
The best way to install pytorch is via the [**pytorch webpage**](http://pytorch.org/)

#### Download GloVe vectors and extract glove.6B.100d.txt into "./data/" folder

`wget http://nlp.stanford.edu/data/glove.6B.zip`

### How to run experiments
To run [CNN char + word emb] + [CNN one layer], change config.py: 
```python
parameters['crf'] = 1
parameters['char_mode']='CNN'
parameters['CNN_params'] = {'kernel_size':[3], 'dilation': [1], 'n_layers':1}
```
To run [LSTM char + word emb] + [CNN 1-4 layers] + CRF, change config.py:
```python
parameters['crf'] = 1
parameters['char_mode']='LSTM'
parameters['CNN_params'] = {'kernel_size':[3,3,3,3], 'dilation': [1,1,1,1], 'n_layers':num_of_layer_from_1_to_4}
```
To run [LSTM char + word emb] + [CNN 3 dialated  layers] + CRF, change config.py:
```python
parameters['crf'] = 1
parameters['char_mode']='LSTM'
parameters['CNN_params'] = {'kernel_size':[3,3,3], 'dilation': [1,2,3], 'n_layers':3}
```
To run [LSTM char + word emb] + [CNN 1 layers] + Softmax, change config.py:
```python
parameters['crf'] = 0
parameters['char_mode']='LSTM'
parameters['CNN_params'] = {'kernel_size':[3], 'dilation': [1], 'n_layers':1}
```

Then just run
`python main.py`