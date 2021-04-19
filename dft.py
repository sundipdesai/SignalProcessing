# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:51:02 2021

@author: sundip desai

Code to help understand the mechanics of 
how discrete fourier transforms (DFT) work.

References:
https://www.allaboutcircuits.com/technical-articles/an-introduction-to-the-discrete-fourier-transform/
https://jackschaedler.github.io/circles-sines-signals/dft_walkthrough.html
"""
import numpy as np
import matplotlib.pyplot as plt

# Generate signal
# Use the form x(t) = A1*sin(omega*t+phi) to generate

#%% Signal Synthesis
Fs = 8 #sample frequency
Ts = 1/Fs #sample period
N = 2000  #length of signal
t=Ts*np.arange(0,N,1) #time vector
phi1=0 #phase shift
phi2=0
w1=1*2*np.pi #1hz signal
w2=3*2*np.pi #3hz signal
A1=1 #amplitudes          
A2=1
noise = np.random.normal(size=len(t)) #mean = 0, variance = 1


# generate signal 
x = noise+(A1*np.sin(w1*t+phi1)+A2*np.cos(w2*t+phi2))

plt.figure(1)
plt.plot(t,x)
plt.ylabel('Amplitude'), plt.xlabel('Time')
plt.grid()
plt.title('Signal Time History')

#%% DFT
# Remarks:
# -------
# Project the sampled signal, x[n] onto 
# the cosine and sine basis vectors
#
# Discretize the basis vectors into the same number of samples as 
# x[n]
#
# We presume x[n] is periodic so it will project onto a basis vector which is periodic. 
# As we sum all contributions from each basis
# vector we increase 'k' or discretized frequency term by 1. k is the same length
# of the number of samples in the sampled signal.
# 
# The output are the DFT coefficients that are used to construct the spectrum

X = np.zeros((N-1),dtype=np.complex_)
for k in np.arange(0,N-1,1): #loop through frequencies (bins)
    print("Bin#: ", k)
    for n in np.arange(0,N-1,1): #loop through signal
        X[k]= X[k] + x[n]*np.exp(-np.complex(0,1)*2*np.pi/N*n*k)

#%%
# Compute magnitudes and plot the spectrum
mag=abs(X) 
bins=np.arange(0,Fs/2,Fs/N) #one-sided spectrum
plt.figure(2)
plt.plot(bins[:-1],mag[0:int(len(mag)/2)])
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')

#%% END















