from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.fftpack
from numpy.fft import rfft

#Import Data
rocketData = pd.read_csv('TestData.csv');
sample_rate = 2500.0
nsamples = len(rocketData['Column_A'])
t = arange(nsamples) / sample_rate

# Create a FIR filter
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
plot((w/pi)*nyq_rate, 20*np.log10(np.abs(h)), linewidth=2)
#plt.xscale('log')
plt.xlim(0,int((width * nyq_rate / 2 + cutoff_hz)*1.2))
plt.ylim(int((-ripple_db)*1.2),0)
xlabel('Frequency (Hz)')
ylabel('Gain dB')
title('Frequency Response. Roll of = ' + str(int(rollOff)) + 'dB/Octave')
grid(True)
# Upper inset plot.
ax1 = axes([0.22, 0.3, .45, .25])
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
grid(True)

#Apply FIR
# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate
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