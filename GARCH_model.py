# Created by ivywang at 2024-01-14
import pandas as pd
import numpy as np
from os import chdir
from arch import arch_model
import matplotlib.pyplot as plt

#S&P 500 data cleaning
sp_data = pd.read_csv('S&P500 2020.csv')
sp_data = sp_data[['Date','Adj Close']]
sp_data['Date'] = pd.to_datetime(sp_data['Date'])
sp_data = sp_data.rename(columns={'Adj Close':'Price'})
sp_data['Return'] = sp_data['Price'].pct_change()*100
sp_data = sp_data.dropna()
sp_data.set_index('Date',inplace = True)

#GARCH Model Fit
basic_gm = arch_model(sp_data['Return'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
# gm_result = basic_gm.fit(update_freq = 4)
# print(gm_result.summary())
# gm_result.plot()
# plt.show()

#GARCH MODEL Forecast
gm_result = basic_gm.fit()
gm_forecast = gm_result.forecast(horizon = 5)
print(gm_forecast.variance[-1:])

