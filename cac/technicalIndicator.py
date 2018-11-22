import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots

def displaymille():
	print("mille")
	
def calculateMovingAverages(series, windows):
	rolling_mean = {}
	for i in range(len(windows)):
		rolling_mean[i] = series.rolling(window=windows[i]).mean()
	return rolling_mean
	
def plotMovingAverages(series, windows, rolling_means, scale=1.96):
	colors = ['g','r','y','b','m']
	plt.figure(figsize=(15,5))
	plt.title("Moving average")
	for i in range(len(rolling_means)):
		plt.plot(rolling_means[i], colors[i], label="mm{}".format(windows[i]))
	
	plt.plot(series[windows[-1]:], label="Actual values")
	plt.legend(loc="upper left")
	plt.grid(True)
	plt.show()	

def calculateBollingers(series, mm = 20, a=2):#ongoing
	bollingers = {}
	rolling_mean = series.rolling(window=mm).mean()
	#not finished
	return mm, bollingers
	
def plotIndicatorOnPrice(series, windows, rolling_means, scale=1.96):#ongoing
	colors = ['g','r','y','b','m']
	plt.figure(figsize=(15,5))
	plt.title("Moving average")
	for i in range(len(rolling_means)):
		plt.plot(rolling_means[i], colors[i], label="mm{}".format(windows[i]))
	
	plt.plot(series[windows[-1]:], label="Actual values")
	plt.legend(loc="upper left")
	plt.grid(True)
	plt.show()
