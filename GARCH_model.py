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

#GARCH Model Forecast
gm_result = basic_gm.fit(disp= 'off')
gm_forecast = gm_result.forecast(horizon = 5)
# print(gm_forecast.variance[-1:])

#Mean Model Specifications - mean uses AR model
constant_mean_result = arch_model(sp_data['Return'],p =1, q = 1, mean = 'constant', vol = 'GARCH')
constant_mean_model = constant_mean_result.fit(disp= 'off')
# print(constant_mean_model.summary())
cmean_vol = constant_mean_model.conditional_volatility

ar_mean_result = arch_model(sp_data['Return'],p =1, q = 1, mean = 'AR', vol = 'GARCH')
ar_mean_model = ar_mean_result.fit(disp= 'off')
# print(ar_mean_model.summary())
armean_vol = ar_mean_model.conditional_volatility

# plt.plot(cmean_vol,color = 'blue', label = 'Constant Mean Volatility')
# plt.plot(armean_vol,color = 'red', label = 'AR Mean Volatility')
# plt.legend(loc = 'upper right')
# plt.show()
# print(np.corrcoef(cmean_vol,armean_vol)[0,1])

#Volatility Models for asymmetric shocks - GJR GARCH & EGARCH
gjr_gm = arch_model(sp_data['Return'],p =1, q = 1, o = 1, vol = 'GARCH', dist = 't' )
gjr_result = gjr_gm.fit(disp = 'off')
# print(gjr_result.summary())
gjrgm_vol = gjr_result.conditional_volatility

egarch_gm = arch_model(sp_data['Return'],p =1, q = 1, o = 1, vol = 'EGARCH', dist = 't' )
egarch_result = egarch_gm.fit(disp = 'off')
# print(egarch_result.summary())
egarch_vol = egarch_result.conditional_volatility

# plt.plot(sp_data['Return'], color = 'grey',alpha = 0.4, label = 'Price Returns')
# plt.plot(gjrgm_vol, color = 'gold', label = 'GJR-GARCH Volatility')
# plt.plot(egarch_vol, color = 'red', label = 'EGARCH Volatility')
# plt.legend(loc = 'upper right')
# plt.show()

# GARCH Rolling Window Forecast
basic_gm = arch_model(sp_data['Return'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
for i in range(30):
    gm_result = basic_gm.fit(first_obs= i+1000, last_obs= i+1500, update_freq = 5)