"""_____________________________________________________________________________________________________________
||                                                                                                             ||
||                 Time Series Forecasting with Recurrent Neuronal Network : generated stairs                  ||
||                                                                                            By Barna et Raoul||
______________________________________________________________________________________________________________"""


#Unlike prior version, test data are generate separately

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
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

#TF Version
print(tf.__version__)


'''---------------------------------------------------------------------------------------------------------
|--------------------------------------- Generating training data -----------------------------------------|
---------------------------------------------------------------------------------------------------------'''
nb_training_data = 209
price = np.zeros(nb_training_data)
price[0] = 2000
mu = 0
sigma = 1
for y in range(1, nb_training_data):
	if np.mod(y,2) == 1:
		price[y] = price[y-1] + 4 + random.normalvariate(mu, sigma)
	else:
		price[y] = price[y-1] - 8 +  random.normalvariate(mu, sigma)
#plot price curve
plt.figure();plt.plot(price, 'b.');plt.show()
#Convert data into array that can be broken up into training "batches" that we will feed into our RNN model.
TS = np.array(price)
print(TS)
#input("Press ENTER to continue...")

#Convert data into array that can be broken up into training "batches" that we will feed into our RNN model. Note the shape of the arrays.
num_periods = 20
f_horizon = 1  #forecast horizon, one period into the future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[f_horizon:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)


print(x_data)
print(y_data)
"""
print (y_batches[0:1])
print (y_batches.shape)
"""
#Pull out our test data
def test_data(series,forecast,num_periods):
    test_x_setup = series[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = series[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY
    
#generate some training data
nb_testing_data = 21
price_test = np.zeros(nb_testing_data)
price_test[0] = 2000
mu = 0
sigma = 1

for y in range(1, nb_testing_data):
	if np.mod(y,2) == 1:
		price_test[y] = price_test[y-1] + 4
	else:
		price_test[y] = price_test[y-1] - 8
#plot price curve
plt.figure();plt.plot(price_test, 'bo');plt.show()

#Convert data into array that can be broken up into training "batches" that we will feed into our RNN model.
TS_test = np.array(price_test)
print(TS_test)
#input("Press ENTER to continue...")

X_test, Y_test = test_data(TS_test,f_horizon,num_periods)

print(X_test)
print(Y_test)


"""
#test data generation
#generate some training data
nb_testing_data = 40
price = np.zeros(nb_testing_data)
price_testing[0] = price[nb_training_data]
for y in range(1, nb_testing_data_data):
	if np.mod(y,10) < 6:
		price_testing[y] = price_testing[y-1] + 5
	else:
		price_testing[y] = price_testing[y-1] - 8

TS_testing = np.array(price_testing)

def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY

X_test, Y_test = test_data(TS_testing,f_horizon,num_periods)
"""
"""
print (X_test.shape)
print (X_test)

print (Y_test.shape)
print (Y_test)
"""



tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs



'''-------------------------------------------------------------------------------------------------------------------
|--------------------------------------------------Model 'n stuff ---------------------------------------------------|
-------------------------------------------------------------------------------------------------------------------'''
inputs = 1            #number of vectors submitted
hidden = 100        #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create "variable" objects (=placeholder)
y = tf.placeholder(tf.float32, [None, num_periods, output])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results

inputs = 1  
output =1

loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables







'''---------------------------------------------------------------------------------------------------------------
|--------------------------------------Training------------------------------------------------------------------|
---------------------------------------------------------------------------------------------------------------'''

epochs = 1000     #number of iterations of training cycles, includes both the FeedFoward and Backpropogation

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)
    
'''------------------------------------------------------------------------------------------------------------
|--------------------------------------------------Predict----------------------------------------------------|
------------------------------------------------------------------------------------------------------------'''
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)


plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")
plt.savefig('/home/johan/miniconda3/envs/venv/Projets/time_series/new_fig.png')
plt.show()
