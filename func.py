import pandas as pd
import numpy as np 

def back_fill(dataframe, axis):
	df = dataframe.copy()
	for i in reversed((range(len(df[axis])))):
		if df.loc[i, axis] == -1:
			df.loc[i, axis] = df.loc[i+1, axis]

	return df

def get_occurances(dataframe, axis):
	res = {}
	for i in dataframe[axis]:
		if i not in res:
			res[i] = 1
		else:
			res[i] += 1

	return np.array(list(dict(sorted(res.items())).items()))

def fourier_transform(dataframe, axis):								
	t = np.arange(0, len(dataframe-1)) 							#the timescale in of the dataframe
	data_t = dataframe[axis].to_numpy() 					  	#Array form of desired column in dataframe

	N = len(t)										      		   #number of datapoints
	fhat = np.fft.fft(data_t, N)							 		#fast fourier transform desired column data
	PSD = fhat * np.conj(fhat) / N  								#multiplying fhat by its conjugate to get real frequency values
	omega = np.fft.fftfreq(N)									   #creating possible frequency values in the range of N
	return np.real(PSD), fhat, omega

"""Note: PSD has infinitesimal complex values due to errors caused by two's complement error 
   while rounding fhat*np.conj(fhat). This is why I take the real part"""


def filter(PSD, fhat, std_threshold):
	freq_mean = np.log2(PSD).mean()								#numpy method to find mean of freq data
	freq_std = np.log2(PSD).std()									#numpy method to find standard deviation
	threshold = freq_mean + std_threshold*freq_std 		   #calculating noise threshold
	indices = np.log2(PSD) > threshold 							#finding valid indeces in the power spectral density

	fhat_filtered = fhat * indices
	return fhat_filtered

"""Note: values relating to the square root of the PSD are used for determining the threshold coefficient
	since the PSD values do not represent the mean and std of the original fhat but its square"""

def ifft_t(fhat, omega, t):
	cn = np.array([x for x in fhat if x!= 0]) / len(fhat)			 		
	w = np.array([omega[i] for i in range(len(fhat)) if fhat[i] != 0])
	#discard 0 frequencies to save time
	
	res = cn[0]
	for i in range(1,len(w)):
		res += cn[i]*np.exp(1j*2*np.pi*w[i]*t)				#This simply converts our coefficients into a fourier series
	return res 