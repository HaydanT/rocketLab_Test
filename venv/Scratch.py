from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from numpy.fft import rfft
from scipy import misc
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, kaiserord, lfilter, firwin, freqz, filtfilt
import csv

#Import Data
plt.close('all')
rocketData = pd.read_csv('TestData.csv');
print(rocketData)

np.random.seed(0)

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



