from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import technicalIndicator as ti

print(tf.__version__)

###############



# read file
filename = '/home/johan/miniconda3/envs/venv/Projets/cac/cac_40_2006_2015_correct.txt'
data = np.loadtxt(filename)

# select column in file (here price at close) and display price vs time plot
price = np.zeros(len(data)-1)
for y in range(len(data)-1):
   price[y] = data[y][5]

#plot price curve
"""plt.figure();plt.plot(price);plt.show()"""


#labelize, display delta histogram, display label histogram
delta = np.zeros(len(data)-1)
labels = np.zeros(len(data)-1)

y = 0
while y < len(data)-2:
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

"""#plot histrograms of delta and labels 
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
"""


######pre-process
  

n = [7,14,21,28,35,42,49,56]

mmCalculated = np.zeros((len(price),len(n),2))

for j in range(60, len(price)):#iterate over every ticks
  mm = ti.calculateMovAvg(price, j, n)#temporary variable
  mmCalculated[j,:,0]=n
  mmCalculated[j,:,1]=mm

print("moving averages : \n", mmCalculated[1001,:,:])

# function that divides dataset into training and testing sets    
def divideDataset(price, r): # price must be a vector containing the whole data and r and 1-r are the fractions of data dedicated to training and testing respectively
   R = int(r*len(data))# index of the last data point of the training set
   train_data = np.zeros(R)  
   test_data = np.zeros(len(data) - R - 1)
   train_labels = np.zeros(R)  
   test_labels = np.zeros(len(data) - R - 1)
   
   for y in range(0, len(data)-1):  
    if y <= R-1:
      train_data[y] = price[y]
    if y >= R:
      test_data[y-R] = price[y]

   return train_data, test_data, R
   
[train_data, test_data, R] = divideDataset(price, 0.8)

print("R = ", R)
print("Data set size : training = ", len(train_data), "; testing = ", len(test_data))

ti.displaymille()

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

