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

#------------------------------------------------
# Create a signal for demonstration.
#------------------------------------------------

sample_rate = 2500.0
nsamples = len(rocketData['Column_A'])
t = arange(nsamples) / sample_rate
x = rocketData['Column_A']


#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 50.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple=ripple_db, width=width)

# The cutoff frequency of the filter.
cutoff_hz = 100.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(numtaps=N, cutoff=cutoff_hz/nyq_rate, window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)

#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

figure(1)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
grid(True)


#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

w, h = freqz(taps, worN=8000)
figure(2)
clf()
plot((w/pi)*nyq_rate, 20*np.log10(np.abs(h)), linewidth=2)
plt.xscale('log')
xlabel('Frequency (Hz)')

ylabel('Gain dB')
title('Frequency Response')
grid(True)

# Upper inset plot.
ax1 = axes([0.22, 0.3, .45, .25])
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
grid(True)

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------

# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate

figure(3)
# Plot the original signal.
plot(t, x)
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

xlabel('t')
grid(True)

show()