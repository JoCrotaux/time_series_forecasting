from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras
tfe = tf.contrib.eager
tf.enable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt

import technicalIndicator as ti

print(tf.__version__)

###############



# read file
filename = '/home/johan/miniconda3/envs/venv/Projets/cac/cac_40_2006_2015_correct.txt'
data = np.loadtxt(filename)

# select column in file (here price at close) and display price vs time plot
price = np.zeros(len(data))
for y in range(len(data)):
   price[y] = data[y][5]

#plot price curve
plt.figure();plt.plot(price);plt.show()


#labelize, display delta histogram, display label histogram
delta = np.zeros(len(data)-1)
labels = np.zeros(len(data)-1)

y = 0
while y < len(data)-1:
  delta[y] = price[y+1] - price[y]
  if delta[y]>-20 and delta[y]<20:#flat
   labels[y]=0
  if delta[y]<-20 and delta[y]>-60:#moderate down movement
   labels[y]=-1
  if delta[y]<-60:#strong down movement
   labels[y]=-2
  if delta[y]>20 and delta[y]<60:#moderate up movement
   labels[y]=1
  if delta[y]>60:#strong up movement
   labels[y]=2
  y += 1

#plot histrograms of delta and labels 
plt.figure()
plt.hist(delta, bins = 50)
plt.xlabel("delta")
_ = plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(labels, bins=5)
plt.xlabel("label")
_ = plt.ylabel("Count")
plt.show()

mu = np.mean(delta)
sigma = np.std(delta)
print('mean : ', mu, '; std: ', sigma)

priceSimulated1 = np.zeros(len(data))
deltaSimulated1 = np.zeros(len(data)-1)

priceSimulated1[0] = price[0]

y = 0
while y < len(data)-1:
  priceSimulated1[y+1] = priceSimulated1[y] + mu + sigma * np.random.randn()
  deltaSimulated1[y] = priceSimulated1[y+1] - priceSimulated1[y]
  y += 1
  
muSimulated1 = np.mean(deltaSimulated1)
sigmaSimulated1 = np.std(deltaSimulated1)
print('mean : ', muSimulated1, '; std: ', sigmaSimulated1)

priceSimulated2 = np.zeros(len(data))
deltaSimulated2 = np.zeros(len(data)-1)
priceSimulated2[0] = price[0]
y = 0
while y < len(data)-1:
  deltaSimulated2[y] = (mu + sigma * np.random.randn()) 
  priceSimulated2[y+1] = priceSimulated2[y] + deltaSimulated2[y]
  
  y += 1

muSimulated2 = np.mean(deltaSimulated2)
sigmaSimulated2 = np.std(deltaSimulated2)
print('mean : ', muSimulated2, '; std: ', sigmaSimulated2)

plt.figure();plt.plot(price, c='b', label = 'Real data');plt.plot(priceSimulated1, c='r', label = 'Simulation 1');
plt.plot(priceSimulated2, c='g', label = 'Simulation 2');plt.legend();plt.title('CAC 40 - Simulation assuming Wiener process');plt.ylabel('Price');
plt.xlabel('Time - daily tick');plt.show()


# function that divides dataset into training and testing sets    
def divideDataset(price, r): # price must be a vector containing the whole data and r and 1-r are the fractions of data dedicated to training and testing respectively
   R = int(r*len(delta))# index of the last data point of the training set
   train_data = np.zeros(R)  
   test_data = np.zeros(len(delta) - R)
   train_labels = np.zeros(R)  
   test_labels = np.zeros(len(delta) - R)
   
   for y in range(len(delta)):  
    if y <= R-1:
      train_data[y] = price[y]
    if y >= R:
      test_data[y-R] = price[y]

   return train_data, test_data, R
   
[train_data, test_data, R] = divideDataset(price, 0.8)

print("R = ", R)
print("Data set size : training = ", len(train_data), "; testing = ", len(test_data))

"""
print(train_data[0])
print(train_data[R-1])
print(delta[R-1])
print(len(test_data))
print(test_data[0])
print(test_data[len(test_data)-1])
"""

"""
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[0],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(5)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model
""" 

"""
model = build_model()
model.summary()

history = model.fit(train_data, train_labels, epochs=10,
                    validation_split=0.2, verbose=0)
"""

