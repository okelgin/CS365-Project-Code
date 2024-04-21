import func
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import kurtosis
from scipy.stats import skew

sns.set_theme(style='darkgrid')

#Preliminary Imputation
sunspots_init = pd.read_csv("sunspot_data.csv")
sunspots_filled = func.back_fill(sunspots_init, 'Number of Sunspots')

#Obtaining statistics
sunspot_mean = sunspots_filled['Number of Sunspots'].mean()
sunspot_std =  sunspots_filled['Number of Sunspots'].std()
sunspots_kurt = kurtosis(sunspots_filled['Number of Sunspots'], fisher=False)
sunspots_skew = skew(sunspots_filled['Number of Sunspots'])
occurances = func.get_occurances(sunspots_filled, 'Number of Sunspots')

#Fourier transform and filtering
PSD, fhat, omega = func.fourier_transform(sunspots_filled, 'Number of Sunspots')
sunspots_filtered = func.filter(PSD, fhat, sunspots_kurt)



#Plotting

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

plt.plot(omega, np.sqrt(PSD), color='b')
plt.ylabel('Square Root PSD')
plt.xlabel('Frequency (1/days)')
plt.show()


# f_plot = sns.relplot(data=sunspots_filled, x='Unnamed: 0', y='Number of Sunspots', kind='line')
# f_plot.ax.set(xlabel='Time in Days Since First Datapoint', ylabel='Number of Sunspots')
# plt.plot(range(len(sunspots_filtered)), sunspots_filtered, color='salmon', label='Filtered Number of Sunpots')
# plt.legend()
# plt.show()