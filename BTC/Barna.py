import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loadData(filepath, column = 5):
	"""
		load price in text file and convert it into numpy aray
	"""
	data = np.loadtxt(filepath)
	price = np.zeros(len(data))
	for y in range(len(data)):
		price[y] = data[y][column]
	return price

def loadCandle(filepath, columns = [2,3,4,5]):
	"""
		load candle in text file and convert it into numpy matrix
		cfr CAC40 2006-2015 daily [open high low close]
	"""
	data = np.loadtxt(filepath)
	candle = np.empty((len(data),4))
	for y in range(len(data)):
		candle[y][0] = data[y][columns[0]]
		candle[y][1] = data[y][columns[1]]
		candle[y][2] = data[y][columns[2]]
		candle[y][3] = data[y][columns[3]]
	return candle

def calculateVariation(np_series, horizon = 1, percent = 0):
	"""
		Return variation between serie in t and t+horizon
		Parameter "percent" : 0 for absolute variaition, 1 for percentage variation
	"""
	variation = np.empty(len(np_series))
	for i in range(horizon,len(np_series)):
		variation[i] = np_series[i] - np_series[i-horizon]
		if bool(percent == 1):
			variation[i] = variation[i]/np_series[i-horizon]*100
	return variation
	
def displayPlot(np_series):
	"""
		Plot a serie
	"""
	plt.figure()
	plt.plot(np_series)
	plt.show()
	
def displayHist(np_series, horizon = 1, bins = 50):
	"""
		Histogram of an array of variations 
	"""
	mean_series = np.mean(np_series)
	std_series = np.std(np_series)
	plt.figure()
	plt.hist(np_series, bins)
	plt.title("Variation for horizon = {}\n".format(horizon))
	plt.xlabel("Mean = {}, std = {}".format(np.around(mean_series, decimals = 2), np.around(std_series, decimals = 2)))
	plt.show()
	
def pointsPivot(np_matrix):
	"""
		Calculate Pivot Points, Resistances and Supports from one tic to the next one
			order in  input np_matrix : [open high low close]
			output np_matrix [SIZE : len(np_matrix) Lines, 7 Columns] : PP, R1, S1, R2, S2, R3, S3
	"""
	PSR = np.zeros((len(np_matrix),7))
	for i in range(1, len(np_matrix)):
		PP = (np_matrix[i-1][1] + np_matrix[i-1][2] + np_matrix[i-1][3])/3
		PSR[i,0] = PP
		PSR[i,1] = 2 * PP - np_matrix[i-1][2]
		PSR[i,2] = 2 * PP - np_matrix[i-1][1]
		PSR[i,3] = PP + np_matrix[i-1][1] - np_matrix[i-1][2]
		PSR[i,4] = PP - np_matrix[i-1][1] + np_matrix[i-1][2]
		PSR[i,5] = np_matrix[i-1][1] + 2 * (PP - np_matrix[i-1][2])
		PSR[i,6] = np_matrix[i-1][2] - 2 * (np_matrix[i-1][1] - PP)
	return PSR#PP, R1, S1, R2, S2, R3, S3

def Bollingers(np_series, mm = 20, a=2):
	"""
		Calculate moving averages of a numpy serie and related bollingers bands
			output np_matrix [SIZE : len(np_matrix) Lines, 3 Columns] : MA, bollinger_up, bollinger_down
	"""
	bollingers = np.zeros((len(np_series),3))
	rolling_std = np.zeros(len(np_series))
	
	window = mm
	
	for i in range(window, len(np_series)):
		bollingers[i,0] = np.mean(np_series[i - window: i])
		rolling_std[i] = np.std(np_series[i - window: i])
		bollingers[i,1] = bollingers[i,0] + a * rolling_std[i]
		bollingers[i,2] = bollingers[i,0] - a * rolling_std[i]
		
	return bollingers

def plotBollingers(np_series, bollingers, window = 20, scale=1.96):
	colors = ['g','r','y','b','m']
	plt.figure(figsize=(15,5))
	plt.title("Bollingers")
	
	plt.plot(bollingers[:,0], colors[0], label="mm{}".format(window))
	plt.plot(bollingers[:,1], colors[1], label="Bollinger sup")
	plt.plot(bollingers[:,2], colors[1], label="Bollinger inf")
	
	plt.plot(np_series, label="Actual values")
	plt.legend(loc="upper left")
	plt.grid(True)
	plt.show()	
