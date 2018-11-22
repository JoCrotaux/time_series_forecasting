import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Barna as bn

filename = '/home/johan/miniconda3/envs/venv/Projets/cac/cac_40_2006_2015_correct.txt'

price = bn.loadData(filename)
bn.displayPlot(price)


bollingers = bn.Bollingers(price)
bn.plotBollingers(price,bollingers)



