"""
	This script aims to study bitcoin's price at close after an "excessive" intraday variation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Barna as bn
import os

dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'BTCUSD_2016_2018.txt')

#BTC plot and variations from one day to the next one
price = bn.loadData(filepath, column = 4)
print(price)
bn.displayPlot(price)
variation = bn.calculateVariation(price, percent = 1)
print(np.mean(variation))
bn.displayHist(variation)

#Main part of the script
candle = bn.loadCandle(filepath, columns = [1,2,3,4])#[open high low close]

satisfied_cond_index = []
variation_cond = []
down = 0
up = 0

loss = 10#percent

for index in range(len(candle)):
	treshold = (1-loss/100)*candle[index, 0]
	c1 = bool(candle[index,2] < treshold)
	if c1:
		satisfied_cond_index = np.append(satisfied_cond_index, [index])
		index = int(index)
		delta = (candle[index, 3] - treshold)/treshold*100
		variation_cond = np.append(variation_cond, [delta])
		if delta > 0:
			up = up + 1
		else:
			down = down + 1
			

		
print(satisfied_cond_index)
print(variation_cond)
print(len(variation_cond))

print('up : ', up/len(variation_cond)*100, ' down : ', down/len(variation_cond)*100)
bn.displayHist(variation_cond)

