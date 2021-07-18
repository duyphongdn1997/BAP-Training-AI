import tensorflow as tf
import numpy as np
import io
from math import exp
import unidecode
import os
import re
import random
import sys
import time
import matplotlib.pyplot as plt
import sys
from keras.utils import plot_model
# tf.enable_eager_execution()

from utils import *
totalTime = time.time()
#############
train_row = read_text('shakespeare_train.txt')
val_row = read_text('shakespeare_valid.txt')
#union vocab
vocab_train = set(train_row)
vocab_val = set(val_row)
vocab = vocab_train.union(vocab_val)

# set character that were found in text to the dict
dict_int = {u:i for i, u in enumerate(vocab)}
dict_char =dict(enumerate(vocab))
#set_char = np.array(vocab)
val_x = c2i(val_row, dict_int)
train_x = c2i(train_row, dict_int)
# seq_len = sys.argv[3]  #100 50
# seq_len = int(seq_len)
seq_len = 50
Ex_1epoch_train = len(train_x) // (seq_len + 1)
Ex_1epoch_val = len(val_x) // (seq_len + 1)

data_train = handle_data(train_x, seq_len) #include input and target
data_val = handle_data(val_x, seq_len)  #include input and target

# Batch size

BATCH_SIZE = 64
iterator_train = Ex_1epoch_train // BATCH_SIZE
iterator_val = Ex_1epoch_val // BATCH_SIZE

BUFFER_SIZE = Ex_1epoch_train+Ex_1epoch_val
data_train = data_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
data_val = data_val.batch(BATCH_SIZE, drop_remainder=True)

########################################################################################################################
#Built The Model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024 #1024 512
rnn_units = int(rnn_units)
cellType  = "LSTM" #"LSTM" , "GRU" , "SimpleRNN"
#cellType  = "RNN"
model = built_model(cellType,vocab_size,embedding_dim,rnn_units,BATCH_SIZE)
model.compile(optimizer = tf.train.AdamOptimizer(),loss=loss)
print(model.summary())
