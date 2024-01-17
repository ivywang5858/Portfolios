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
gm_result = basic_gm.fit(update_freq = 4,disp='off')
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
first_obs = 0
end_obs = 100
forecasts= {}
forecasts_ex= {}
basic_gm = arch_model(sp_data['Return'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
#Rolling Window to forecast variance
#Fixed Window
for i in range(300):
    gm_result = basic_gm.fit(first_obs= i+first_obs, last_obs= i+end_obs, update_freq = 5, disp = 'off')
    temp_result = gm_result.forecast(horizon = 1).variance
    fcast = temp_result.iloc[0]
    forecasts[fcast.name] = fcast
forecast_var = pd.DataFrame(forecasts).T

#Expanding Window
for i in range(300):
    gm_result_ex = basic_gm.fit(first_obs= first_obs, last_obs= i+end_obs, update_freq = 5, disp = 'off')
    temp_result_ex = gm_result_ex.forecast(horizon = 1).variance
    fcast_ex = temp_result_ex.iloc[0]
    forecasts_ex[fcast_ex.name] = fcast_ex
forecast_var_ex = pd.DataFrame(forecasts_ex).T

#Plot the forecast variance
# plt.plot(forecast_var, color = 'red')
# plt.plot(sp_data.Return['2020-05-27':'2021-08-03'],color='green')
# plt.show()

#Plot both volatility forecast - expending and fixed window
vol_fixedwin = np.sqrt(forecast_var)
vol_expandwin = np.sqrt(forecast_var_ex)
plt.plot(vol_fixedwin, color = 'red')
plt.plot(vol_expandwin, color = 'blue')
plt.plot(sp_data['Return']['2020-05-27':'2021-08-03'],color='green')
plt.show()
