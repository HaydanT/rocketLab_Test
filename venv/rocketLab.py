import scipy
import scipy.fftpack
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, kaiserord, lfilter, firwin, freqz, filtfilt
import csv
from numpy.fft import rfft


def fftdB(array, bins=10000, sampleRate=31250, chart=False, real=False):
    freq = 1.0 / sampleRate
    length = len(array);
    if (length < bins) : bins = length;
    if (real):
        yf = scipy.fftpack.rfft(array)
        yfdBtemp = np.abs(yf) * 2
        yfdB = 20 * np.log10(yfdBtemp)
        yChart = 2.0 / bins * np.abs(yfdB[1:int(bins / 2 + 1)])
        if (chart):
            xChart = np.linspace(1.0, 1.0 / (2 * freq) + 1, bins / 2.0)
    else:
        yf = scipy.fftpack.fft(array)
        yfdBtemp = np.abs(yf) * 2
        yfdB = 20 * np.log10(yfdBtemp)
        yChart = 2.0 / bins * np.abs(yfdB[1:int(bins // 2) + 1])
        if (chart):
            xChart = np.linspace(1.0, 1.0 / (2.0 * freq) + 1, bins / 2.0)
    if (chart):
        xf = np.linspace(1.0, 1.0 / (2.0 * sampleRate) + 1, bins / 2.0)
        fig, ax = plt.subplots()
        ax.plot(xChart, 20 * np.log10(yChart))
        maxLog = np.amax(yChart)
        maxFreq = np.where(yChart == maxLog)
        maxFreqData = xChart[maxFreq[0][0]]
        plt.annotate('Strongest Component at : ' + str(round(maxFreqData,1)) + "Hz", xy=(.30, .15), xycoords='figure fraction')
        plt.show()
        print(str(max(xChart)))
        print(str(len(yChart)))
    return yChart

#Import Data
plt.close('all')
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

#FIR Filter - ToDo Review this!
fs = 2500  # Sampling frequency
fc = 100  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
t = rocketData['Time (s)']
for i in range(1, 4):
    signalc = rocketData[rocketData.columns[i]]
    plt.plot(t, signalc, label=rocketData.columns[i])
    b, a = signal.butter(5, w, 'low')
    output = scipy.signal.filtfilt(b, a, signalc)
    plt.plot(t, output, label='filtered')
    plt.legend()
    plt.show()

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
plt.annotate('The equation is B = ' + str(round(slope, 3)) + '*A + ' + str(round(inter,3)) , xy=(.4, .30), xycoords='figure fraction')
plt.annotate('R^2 of ' + str(round(R2,5)), xy=(.40, .25), xycoords='figure fraction')
plt.show();

#Butterworth Filter - - ToDo Review This!
# 1st Order so shoulder is about at about ~50Hz (1 order down)
dt = 0.0004  # sampling interval
Fs = 1 / dt  # sampling frequency
t = rocketData['Time (s)']
s = rocketData['Column_C']

fig, axes = plt.subplots(nrows=2, ncols=1)

# plot time signal:
axes[0].set_title("Column_C")
axes[0].plot(t, s, color='C0')
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Amplitude")

axes[1].set_title("Log. Magnitude Spectrum")
y, x, waste = axes[1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

fig.tight_layout()
plt.show()

#Butterworth Filter
# 1st Order so shoulder is about at about ~50Hz (1 order down)
filter_order = 1
frequency_cutoff = 50
sampling_frequency = 2500
# Create the filter
b, a = butter(N=filter_order, Wn=frequency_cutoff, btype='low', analog=False, output='ba', fs=sampling_frequency)

#What does the filter look like? The large dB drop at exactly 50 is from the signal there being scaled
dataLen = len(rocketData['Column_C'])
impulse = np.zeros(dataLen)
impulse[int(dataLen/2)] = 1
imp_ff = signal.filtfilt(b, a, impulse)
plt.semilogx(20*np.log10(np.abs(rfft(imp_ff))))
plt.ylim(-100, 20)
plt.grid(True, which='both')
plt.title('Butter Filter with order : ' + str(filter_order) + ' and cut at : ' + str(frequency_cutoff))
plt.show()

# Apply the filter to data
filtered = filtfilt(b, a, rocketData['Column_C'])
# Plot
plt.plot(rocketData['Time (s)'], rocketData['Column_C'])
plt.plot(rocketData['Time (s)'], filtered)
plt.title("Butter Filter, Cutoff at " + str(frequency_cutoff) + "Hz")
plt.legend(['Original','Filtered'])
plt.show()

#Seeing what butters doing:
dt = 0.0004  # sampling interval
Fs = 1 / dt  # sampling frequency
t = rocketData['Time (s)']
s = rocketData['Column_C']
fig, axes = plt.subplots(nrows=2, ncols=1)
# plot FFT's:
axes[0].set_title("Log. Magnitude Spectrum")
resp, respFreq, trash = axes[0].magnitude_spectrum(s, Fs=Fs, scale='linear', color='C1')
axes[0].set_xscale('log')
maxValue = np.amax(resp[2:])
maxValueLoc = np.where(resp == maxValue)
maxFreqVal = respFreq[maxValueLoc[0][0]]
plt.annotate('Max magnitude at frequency : ' + str(int(maxFreqVal)) + 'Hz with value : ' + str(round(maxValue)) , xy=(.30, .25), xycoords='figure fraction')

maxFreq = (np.abs(respFreq - 500)).argmin()
maxFreqData = respFreq[maxFreq]
maxLog = resp[maxFreq]
plt.annotate('Cut off frequency : ' + str(frequency_cutoff), xy=(.30, .30), xycoords='figure fraction')

axes[1].set_title("Log. Magnitude Spectrum (Buttered)")
resp2, respFreq2, trash2 = axes[1].magnitude_spectrum(filtered, Fs=Fs, scale='linear', color='C1')
axes[1].set_xscale('log')
newValue = resp2[maxFreq]

fig.tight_layout()
plt.show()