import scipy
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

#Import Data
rocketData = pd.read_csv('TestData.csv');
print(rocketData)

#plot of all columns against time
for i in range(1, 4):
    rocketData.plot(x = 'Time (s)', y = i, title = 'Raw Time Series')
#    plt.show()

rolling50 = rocketData.rolling(window=50).mean()
for i in range(1, 4):
    rolling50.plot(x = 'Time (s)', y = i, title = 'Rolling Average 50')
#    plt.show()

#FIR FILTER HERE


#rLMean = rocketData[rocketData.columns[1]].mean()
#rocketData['Mean'] = rLMean
#print(rocketData)

#Control chart
for i in range(1, 4):
    rLMean = rocketData[rocketData.columns[i]].mean()
    rocketData['Mean'] = rLMean
    rLControl = rocketData[rocketData.columns[i]].std() * 2;
    rocketData['Mean'] = rLMean;
    rocketData['Lower'] = rLMean - rLControl;
    rocketData['Upper'] = rLMean + rLControl;
    rocketData.plot(x = 'Time (s)', y = [rocketData.columns[i], 'Mean', 'Lower', 'Upper'], title = 'Control Lines', color = ['black', 'blue', 'red', 'red'])
    plt.show()

