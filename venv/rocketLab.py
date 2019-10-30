import scipy
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

#plot of all columns against time
rocketData = pd.read_csv('TestData.csv');
print(rocketData)
print(rocketData['Time (s)'])
print(rocketData['Column_A'])
for i in range(1, 4):
    rocketData.plot(x = 'Time (s)', y = i, title = 'Raw Time Series')
    plt.show()

rolling50 = rocketData.rolling(window=50).mean()
for i in range(1, 4):
    rolling50.plot(x = 'Time (s)', y = i, title = 'Rolling Average 50')
    plt.show()

