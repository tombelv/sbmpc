import matplotlib.pyplot as plt
import numpy as np


nogains = np.load('nogains.npy')
gains = np.load('gains.npy')

time_vect = 0.02*np.arange(gains.shape[0])
plt.plot(time_vect, nogains[:, 0], label='No Gains')
plt.plot(time_vect, gains[:, 0], label='With Gains')
plt.hlines(0.5, 0.0, time_vect[-1], colors='k', linestyles='dashed')

plt.legend()
plt.show()
