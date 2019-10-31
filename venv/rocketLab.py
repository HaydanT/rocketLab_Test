import scipy
import scipy.fftpack
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, kaiserord, lfilter, firwin, freqz, filtfilt
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import csv
from numpy.fft import rfft

#-----------------
#5a) - Import Data
#-----------------
plt.close('all')
rocketData = pd.read_csv('TestData.csv');
sample_rate = 2500.0
print(rocketData)

#---------------------------
#5bi) - Plot all time series
#---------------------------
for i in range(1, 4):
    rocketData.plot(x = 'Time (s)', y = i, title = 'Raw Time Series')
    plt.show()

#-------------------------------
#5bii) - 50 POint moving average
#-------------------------------
rolling50 = rocketData.rolling(window=50).mean()
for i in range(1, 4):
    rolling50.plot(x = 'Time (s)', y = i, title = 'Rolling Average 50')
    plt.show()

#-------------------
#5biii) - FIR Filters
#-------------------
# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0
# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.
transition = 200.0
width = (transition*2)/nyq_rate
# The desired attenuation in the stop band, in dB.
ripple_db = 200.0
# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple=ripple_db, width=width)
# The cutoff frequency of the filter.
cutoff_hz = 100.0
# Roll off = Average drop per octave in tansition region
rollOff = (cutoff_hz * (ripple_db-3) / transition)
# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(numtaps=N, cutoff=cutoff_hz/nyq_rate, window=('kaiser', beta))

# Plot the FIR filter coefficients.
figure(1)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
grid(True)

# Plot the magnitude response of the filter.
w, h = freqz(taps, worN=8000)
figure(2)
clf()
plot((w/np.pi)*nyq_rate, 20*np.log10(np.abs(h)), linewidth=2)
#plt.xscale('log')
plt.xlim(0,int((width * nyq_rate / 2 + cutoff_hz)*1.2))
plt.ylim(int((-ripple_db)*1.2),0)
xlabel('Frequency (Hz)')
ylabel('Gain dB')
title('Frequency Response. Roll of = ' + str(int(rollOff)) + 'dB/Octave')
grid(True)
# Upper inset plot.
ax1 = axes([0.22, 0.3, .45, .25])
plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
grid(True)

#Apply FIR
# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate
nsamples = len(rocketData['Column_A'])
t = np.arange(nsamples) / sample_rate
figure(3)
for i in range(1, 4):
    x = rocketData[rocketData.columns[i]]
    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)
    # Plot the original signal.
    plot(t, x, label='Raw')
    # Plot the filtered signal, shifted to compensate for the phase delay.
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    dataEnd = len(filtered_x) - N
    plot(t[N-1:]-delay, filtered_x[N-1:], 'g', label='filtered')
    title('FIR processed data : ' + rocketData.columns[i])
    xlabel('Time (s)')
    plt.gca().legend(('Raw Data','Filtered Data'))
    grid(True)
    show()

#---------------------
#5biv) - Control Chart
#---------------------
for i in range(1, 4):
    rLMean = rocketData[rocketData.columns[i]].mean()
    rocketData['Mean'] = rLMean
    rLControl = rocketData[rocketData.columns[i]].std() * 2;
    rocketData['Mean'] = rLMean;
    rocketData['Lower'] = rLMean - rLControl;
    rocketData['Upper'] = rLMean + rLControl;
    rocketData.plot(x = 'Time (s)', y = [rocketData.columns[i], 'Mean', 'Lower', 'Upper'], title = 'Control Lines', color = ['black', 'blue', 'red', 'red'])
    plt.show()

#-------------------------
#5c) - Linear Regression
#-------------------------
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

#-------------------------
#5d) - Butterworth Filter
#-------------------------
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
plt.show()