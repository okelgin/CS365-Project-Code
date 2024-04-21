import func
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import kurtosis
from scipy.stats import skew

sns.set_theme(style='darkgrid')

sunspots_init = pd.read_csv('sunspot_data.csv')
sunspots_filled = func.back_fill(sunspots_init, 'Number of Sunspots')


sunspot_mean = sunspots_filled['Number of Sunspots'].mean()
sunspot_std =  sunspots_filled['Number of Sunspots'].std()
sunspots_kurt = kurtosis(sunspots_filled['Number of Sunspots'], fisher=False)
sunspots_skew = skew(sunspots_filled['Number of Sunspots'])

occurances = func.get_occurances(sunspots_filled, 'Number of Sunspots')

# init_plot = sns.relplot(data=sunspots_filled, x='Unnamed: 0', y='Number of Sunspots', kind='line')
# init_plot.ax.set(xlabel='Time in Days Since First Datapoint', ylabel='Number of Sunspots')
# plt.axhline(y = sunspot_mean, color ='red', label='Mean')
# plt.axhline(y = sunspot_mean - sunspot_std, color ='orange', label='Standart Deviation')
# plt.axhline(y = sunspot_mean + sunspot_std, color ='orange')
# plt.legend()
# plt.show()


# plt.plot(occurances[:,0], occurances[:,1])
# plt.axvline(x = sunspot_mean, color ='red', label='Mean')
# plt.axvline(x = sunspot_mean + sunspots_kurt*sunspot_std, color ='orange', label='Kurtosis')
# plt.legend()
# plt.show()
# print(sunspots_skew)
# print(sunspots_kurt)

PSD, fhat, omega = func.fourier_transform(sunspots_filled, 'Number of Sunspots')
sunspots_filtered = func.filter(PSD, fhat, sunspots_kurt)
plt.plot(range(len(sunspots_filtered)), sunspots_filtered)
plt.show()


# t = np.arange(0, len(sunspots_filled-1))                          #The n'th day since the first data point
# num_sunspots_t = sunspots_filled['Number of Sunspots'].to_numpy() #Array form of the number of sunspots as a function of t

# N = len(t)										      		      #number of datapoints
# fhat = np.fft.fft(num_sunspots_t, N)							  #fast fourier transform of sunspot data
# PSD = fhat * np.conj(fhat) / N   								  #multiplying fhat by its conjugate to get real frequency values
# omega = (1/N)*np.arange(N)									 	  #creating possible frequency values in the range of N


# freq_mean = np.real(PSD).mean()								   	  #numpy method to find main
# freq_std = np.real(PSD).std()									  #numpy method to find standard deviation
# threshold = freq_mean + 4*freq_std				
# indices = np.real(PSD) > threshold

# #Note: PSD has infinitesimal complex values due to errors caused by two's complement
# #while rounding fhat*np.conj(fhat). This is why I take the real part

# fhat_filtered = fhat * indices						#zero all the frequencies that are not outliers
# num_sunspots_filtered = np.fft.ifft(fhat_filtered)	#inverse fourier transform


# plt.plot(t, num_sunspots_t, color='b', label='sunspot data')
# plt.plot(t, num_sunspots_filtered, color='orange', label='transformed data', linewidth=2)
# plt.xlabel('Time in days')
# plt.ylabel('Number of sunspots')
# plt.legend()
# plt.show()
