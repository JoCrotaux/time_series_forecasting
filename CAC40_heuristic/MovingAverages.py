import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loadData(filename):
	data = np.loadtxt(filename)
	price = np.zeros(len(data))
	for y in range(len(data)):
		price[y] = data[y][5]
	return price

def loadCandle(filename):
	data = np.loadtxt(filename)
	candle = np.zeros([len(data)][4])
	for y in range(len(data)):
		price[y] = data[y][5]
	return price

def calculateVariation(np_series, horizon = 1):
	variation = np.zeros(len(np_series))
	for i in range(horizon,len(np_series)):
		variation[i] = np_series[i] - np_series[i-horizon]
	return variation
	
def displayPlot(np_series):
	plt.figure()
	plt.plot(np_series)
	plt.show()
	
def displayHist(np_series, horizon = 1, bins = 50):
	mean_series = np.mean(np_series)
	std_series = np.std(np_series)
	plt.figure()
	plt.hist(np_series, bins)
	plt.title("Variation for horizon = {}\n".format(horizon))
	plt.xlabel("Mean = {}, std = {}".format(np.around(mean_series, decimals = 2), np.around(std_series, decimals = 2)))
	plt.show()

filename = '/home/johan/miniconda3/envs/venv/Projets/cac/cac_40_2006_2015_correct.txt'
price = loadData(filename)
displayPlot(price)
variation = calculateVariation(price)
displayHist(variation)

f_horizon = 5
window = [7, 14, 21]
satisfied_cond_index = []


for index in range(window[-1], len(price) - f_horizon):
	mm1 = np.mean(price[index - window[0]: index])
	mm2 = np.mean(price[index - window[1]: index])
	mm3 = np.mean(price[index - window[2]: index])
	c1 = bool(mm1 > mm2)
	#c1 = mm1>mm2
	#momentum1 = price[index] - price[index - 1]
	c2 = bool(mm2 > mm3)
	if c1 and c2:
		satisfied_cond_index = np.append(satisfied_cond_index, [index])


"""
mm = [[],[]]

for index in range(window[-1], len(price)- f_horizon):
	for i in range(len(window)):
		temp_mm = np.mean(price[index - window[i]: index])
		mm[i] = np.append(mm[i], [temp_mm])


print(mm[0][-1])
print(np.shape(mm))
"""	

#print(satisfied_cond_index)
#print(len(satisfied_cond_index))

variation_cond = []		

down = 0
up = 0
flat = 0

for index in satisfied_cond_index:
	i = int(index)
	delta = price[i + f_horizon] - price[i]
	variation_cond = np.append(variation_cond, [delta])
	if delta > 0:
		up = up + 1
	else:
		down = down + 1
		

#print(variation_cond)
print(len(variation_cond))

displayHist(variation_cond, horizon = f_horizon)
print('up : ', up/len(variation_cond)*100, ' down : ', down/len(variation_cond)*100)

