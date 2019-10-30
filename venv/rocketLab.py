import scipy
import scipy.fftpack
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


#Control chart
for i in range(1, 4):
    rLMean = rocketData[rocketData.columns[i]].mean()
    rocketData['Mean'] = rLMean
    rLControl = rocketData[rocketData.columns[i]].std() * 2;
    rocketData['Mean'] = rLMean;
    rocketData['Lower'] = rLMean - rLControl;
    rocketData['Upper'] = rLMean + rLControl;
    rocketData.plot(x = 'Time (s)', y = [rocketData.columns[i], 'Mean', 'Lower', 'Upper'], title = 'Control Lines', color = ['black', 'blue', 'red', 'red'])
#    plt.show()

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
plt.annotate('The equation is B = ' + str(round(slope, 3)) + '*A + ' + str(round(inter,3)) , xy=(.4, .30), xycoords='figure fraction')
plt.annotate('R^2 of ' + str(round(R2,5)), xy=(.40, .25), xycoords='figure fraction')
#plt.show();

#FFT
colCFFTtmp = scipy.fftpack.fft(rocketData['Column_C'])
colCPStmp = np.abs(colCFFTtmp) ** 2
colCFFT = scipy.fftpack.fftfreq(len(colCPStmp), 1/2500)
i = colCFFT > 0
fig, ax = plt.subplots(1, 1)
logOut = 10 * np.log10(colCPStmp[i])
ax.plot(colCFFT[i], logOut)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB)')
maxLog = np.amax(logOut)
maxFreq = np.where(logOut == maxLog)
maxFreqData = colCFFT[maxFreq[0][0]]
plt.annotate('Strongest Component at : ' + str(round(maxFreqData,1)) + "Hz", xy=(.30, .15), xycoords='figure fraction')


fig.show()

#logOut = 10 * np.log10(colCPStmp[i])

print(logOut)
print(maxLog)

print()


#print(colCFFT)