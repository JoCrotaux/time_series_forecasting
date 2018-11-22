import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Barna as bn

filename = '/home/johan/miniconda3/envs/venv/Projets/cac/cac_40_2006_2015_correct.txt'
price = bn.loadData(filename)
bn.displayPlot(price)
variation = bn.calculateVariation(price)
bn.displayHist(variation)

f_horizon = 7
window = 20
satisfied_cond_index = []

candle = bn.loadCandle(filename)

for index in range(window, len(candle) - f_horizon):
	highWindow = np.amax(price[index - window: index])
	highDay = candle[index][1]
	closeDay = price[index]
	c1 = bool(highDay - 10 > highWindow)
	c2 = bool(closeDay < highWindow)
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

bn.displayHist(variation_cond, horizon = f_horizon)
print('up : ', up/len(variation_cond)*100, ' down : ', down/len(variation_cond)*100)






