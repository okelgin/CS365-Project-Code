import func
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy as sci
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

sns.set_theme(style='darkgrid')

#Preliminary Imputation
sunspots_init = pd.read_csv("sunspot_data.csv")
sunspots_filled = func.back_fill(sunspots_init, 'Number of Sunspots')

#Obtaining statistics
sunspot_mean = sunspots_filled['Number of Sunspots'].mean()
sunspot_std =  sunspots_filled['Number of Sunspots'].std()
sunspots_kurt = sci.stats.kurtosis(sunspots_filled['Number of Sunspots'], fisher=False)
sunspots_skew = sci.stats.skew(sunspots_filled['Number of Sunspots'])
occurances = func.get_occurances(sunspots_filled, 'Number of Sunspots')

#Fourier transform and filtering
PSD, fhat, omega = func.fourier_transform(sunspots_filled, 'Number of Sunspots')
fhat_filtered = func.filter(PSD, fhat, sunspots_kurt)
sunspots_filtered = func.ifft_t(fhat_filtered, omega, np.arange(0, len(fhat_filtered)))

#Plotting Commented in order to avoid overlapping images

# init_plot = sns.relplot(data=sunspots_filled, x='Unnamed: 0', y='Number of Sunspots', kind='line')
# init_plot.ax.set(xlabel='Time in Days Since First Datapoint', ylabel='Number of Sunspots')
# plt.axhline(y = sunspot_mean, color ='red', label='Mean')
# plt.axhline(y = sunspot_mean - sunspot_std, color ='orange', label='Standart Deviation')
# plt.axhline(y = sunspot_mean + sunspot_std, color ='orange')
# plt.legend()
# plt.show()

# plt.plot(occurances[:,0], occurances[:,1])
# plt.axvline(x = sunspot_mean, color ='red', label='Mean')
# plt.axvline(x = sunspot_mean + sunspots_kurt*sunspot_std, color ='orange', label='Kurtosis times std')
# plt.legend()
# plt.show()

# plt.plot(omega, np.log2(PSD), color='b')
# plt.ylabel('log2(PSD)')
# plt.xlabel('Frequency (1/days)')
# plt.show()

# plt.plot(omega, np.sqrt(PSD), color='b')
# plt.ylabel('Square Root PSD')
# plt.xlabel('Frequency (1/days)')
# plt.show()


# f_plot = sns.relplot(data=sunspots_filled, x='Unnamed: 0', y='Number of Sunspots', kind='line')
# f_plot.ax.set(xlabel='Time in Days Since First Datapoint', ylabel='Number of Sunspots')
# plt.plot(range(len(sunspots_filtered)), sunspots_filtered, color='salmon', label='Filtered Number of Sunpots')
# plt.legend()
# plt.show()

# acf = plot_acf(sunspots_filled['Number of Sunspots'])
# pacf = plot_pacf(sunspots_filled['Number of Sunspots'])


# idx = np.argmax(fhat[1:])
# w = omega[idx] 											
# """Turns out to be 0.000257738951138121"""
# t = np.arange(0, len(sunspots_filled), 1)
# plt.plot(t, sunspots_filled["Number of Sunspots"], color = 'b', label='Original Data')
# plt.plot(t, sunspots_filtered, color = 'salmon', label='Filtered Sunspots(t)')
# plt.plot(t, 50 + 50*np.sin(w*t+12000), color = 'white', label='Most dominant oscillation')
# plt.legend()
# plt.show()


# diff = sunspots_filled['Number of Sunspots'].to_numpy() - sunspots_filtered
# acf = plot_acf(diff, label= 'Autocorrelation of Residual Difference', color= 'salmon')
# # pacf = plot_pacf(diff)
# # plt.plot(range(len(fhat)), diff, label= 'Difference of Filtered and Actual Data')
# plt.legend()
# plt.show()


