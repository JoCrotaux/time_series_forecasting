import os

"""
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
"""
import numpy as np
import matplotlib.pyplot as plt

import Barna as bn

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname,'cac/cac_40_2006_2015_correct.txt')
price = bn.loadData(filename)
#bn.displayPlot(price)
variation = bn.calculateVariation(price)
#bn.displayHist(variation)

candle = bn.loadCandle(filename)

n = len(candle)
print('number of ticks :', n)

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = price[train_start: train_end]
data_test = price[test_start: test_end]

print('number of ticks :', n)
print('Train = ', len(data_train), 'Test = ', len(data_test))

###scaler

n_inputs = 0
n_steps = 20



n_neurons = n_inputs + 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))



basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run()
	outputs_val = outputs.eval(feed_dict={X: X_batch})









