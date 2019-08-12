
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import pywt
import pywt.data

# import torch
# import torchvision

###########################################################
####################### Load signal #######################
###########################################################


# Sampling frequency
Fs=1e4					# [Hz]

# Sampling time period
Ts= 1/Fs					# [s]

# Angular Velocity (It should later be calculated from the tachometer signal)
ang_vel=35			# [Hz]

# Rotation period
T=1/ang_vel			# [s]

# Test condition can be one of these:
# Run Up, Steady Speed, Shut Down
test_cond='Run Up'

# Loading the raw signal
signal=sio.loadmat("signal.mat")

# Assigning the number of samples to variable L
L=len(signal['tacho'][0])

# Constructing the time vector
time=np.arange(L)*Ts

# Constructing the time vector for one revolution
time_one_period=np.arange(0,T,Ts)

# Pre-allocating the variable s_h to contain the raw signals of 4 sensors
s_h=np.zeros((4,L))

# Assigning the signal of each sensor to each row of the variable s_h
s_h[0,0:]=signal['sensor1'][0]
s_h[1,0:]=signal['sensor2'][0]
s_h[2,0:]=signal['sensor3'][0]
s_h[3,0:]=signal['tacho'][0]

# Plotting the raw signals
fig_raw=plt.figure(figsize=(12,6))
for i in range(3):
    ax_raw=fig_raw.add_subplot(3,1,i+1)   
    ax_raw.plot(time,s_h[i,:])
plt.show()

#################################################################
####################### SIGNAL SEPARATION #######################
#################################################################

# Length of each sub-signal
m=4000

# Assigning the signal of the first accelerometer to variable s1_h
s1_h=s_h[0,0:]

# Calculating the number of sub-signals
n_h=math.floor(len(s1_h)/m)

# Reshaping the raw signal into (n_h) subsignals each with (m) samples
x_h=s1_h.reshape((n_h,m))


####################################################################
####################### SIGNAL DECOMPOSITION #######################
####################################################################


# Single level decomposition
# Decompose sub-signals to detail and approximate
coeffs=pywt.dwt(x_h,'db4')

# L: approximate    H: detail
L,H=coeffs


# Plotting a sample decomposed (detail and approximate) sub-signal 
fig_coef=plt.figure(figsize=(12,4))
for i, a in enumerate([L[6],H[6]]):
    ax_coef=fig_coef.add_subplot(2,1,i+1)
    ax_coef.plot(a)
fig_coef.tight_layout()
plt.show()




# Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# coeffs2 = pywt.dwt2(original, 'bior1.3')
# LL, (LH, HL, HH) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
#plt.show()


