#full tutorial available at https://mapr.com/blog/deep-learning-tensorflow/
#code available at https://github.com/JustinBurg/TensorFlow_TimeSeries_RNN_MapR/blob/master/RNN_Timeseries_Demo.ipynb


#What are we working with?
import sys
print(sys.version)

#Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
#%matplotlib inline
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

#TF Version
print(tf.__version__)

# read file
filename = '/home/johan/miniconda3/envs/venv/Projets/cac/cac_40_2006_2015_correct.txt'
data = np.loadtxt(filename)

# select column in file (here price_30 at close) and display price_30 vs time plot
nb_data = len(data)
price_30 = np.zeros(int(nb_data/30)+1)

print('nbr de mois : ', len(price_30))
z=0
for y in range(nb_data):
	if y%30 == 0:
		price_30[z] = data[y][5]
		z +=1

#plot price_30 curve
plt.figure();plt.plot(price_30);plt.show()

#Convert data into array that can be broken up into training "batches" that we will feed into our RNN model. Note the shape of the arrays.

TS = np.array(price_30)
print(TS)
num_periods = 10
f_horizon = 1  #forecast horizon, one period into the future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[f_horizon:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)
print(x_data)
print(y_data)



#Pull out our test data
def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods )


tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs

      #number of periods per vector we are using to predict one period ahead
inputs = 1            #number of vectors submitted
hidden = 100          #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create variable objects
y = tf.placeholder(tf.float32, [None, num_periods, output])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables

epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)
    
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    #print(y_pred)


plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "b.", markersize=10, label="Actual")
#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")
plt.show()
len(y_pred)
discrepancies = pd.Series(np.ravel(Y_test)) - pd.Series(np.ravel(y_pred))
MAE = pd.Series.mean(pd.Series.abs(discrepancies))
STD_AE = pd.Series.std(pd.Series.abs(discrepancies))

print('MAE = ', MAE,' STD_AE = ', STD_AE)

plt.title("Difference between forecast and actual", fontsize=14)
plt.hist(discrepancies)
plt.ylabel("Occurence")
plt.show()
