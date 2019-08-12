
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import pywt
import pywt.data

# import torch
# import torchvision


#### Load signal ####

original = pywt.data.camera()

Fs=1e4					# Sampling frequency (Hz)
Ts= 1/Fs					# Sampling time period (s)
ang_vel=35			# Angular Velocity (Hz)
T=1/ang_vel			# Rotation period (s)


# Run Up, Steady Speed, Shut Down
test_cond='Run Up'
#sio.loadmat('signal.mat')
signal=sio.loadmat("signal.mat")
#print(signal.keys())
L=len(signal['tacho'][0])
time=np.arange(L)*Ts
time_one_period=np.arange(0,T,Ts)
#print(len(time_one_period))

s1_h=signal['sensor1'][0]
s2_h=signal['sensor2'][0]
s3_h=signal['sensor3'][0]
tacho_h=signal['tacho'][0]


#plt.show()

#### SIGNAL SEPARATION ####

m=5000

n_h=math.floor(len(s1_h)/m)

print(n_h)

x_h=s1_h.reshape((n_h,m))

#print(x_h.shape)

#print(len(x_h[0]))


coeffs=pywt.dwt(x_h,'db4')

#print(len(coeffs[1][29]))
#print(coeffs[1][29][999])
#print(dir(coeffs))
#print(np.ndarray.size(coeffs))

L,H=coeffs

fig_raw=plt.figure(figsize=(12,6))
for i in range(3):
    ax_raw=fig_raw.add_subplot(3,1,i+1)
    ax_raw.plot(time,s1_h)
#fig1.show()

L,H=coeffs

fig_coef=plt.figure(figsize=(12,4))
for i, a in enumerate([L[10],H[10]]):
    ax_coef=fig_coef.add_subplot(2,1,i+1)
    ax_coef.plot(a)

fig_coef.tight_layout()
plt.show()




# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
#plt.show()


