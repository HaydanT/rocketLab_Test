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
    plt.show()

rolling50 = rocketData.rolling(window=50).mean()
for i in range(1, 4):
    rolling50.plot(x = 'Time (s)', y = i, title = 'Rolling Average 50')
    plt.show()


#FIR FILTER HERE


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

#linear regression
A = rocketData['Column_A'];
B = rocketData['Column_B'];
denom = A.dot(A) - A.mean() * A.sum();
slope = (A.dot(B) - B.mean() * A.sum()) / denom;
inter = (B.mean() * A.dot(A) - A.mean() * A.dot(B)) / denom
rocketData['APredB'] = slope*A + inter;
res = B - rocketData['APredB']
resAve = B - res.mean();
R2 = 1 - res.dot(res) / resAve.dot(resAve);
plt.scatter(A,B) #Doesn't look linear
plt.plot(A,rocketData['APredB'], color = 'red')
plt.annotate('figure pixels', xy=(10, 10), xycoords='figure pixels')
plt.annotate('The equation is B = ' + str(round(slope, 3)) + '*A + ' + str(round(inter,3)) , xy=(.4, .30), xycoords='figure fraction')
plt.annotate('R^2 of ' + str(round(R2,5)), xy=(.40, .25), xycoords='figure fraction')
plt.show();