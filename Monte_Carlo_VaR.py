# Created by ivywang at 2024-01-14
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# S&P 500 data cleaning
sp_data = pd.read_csv('S&P500 2020.csv')
sp_data = sp_data[['Date','Adj Close']]
sp_data['Date'] = pd.to_datetime(sp_data['Date'])
sp_data = sp_data.rename(columns={'Adj Close':'Price'})
sp_data['Return'] = sp_data['Price'].pct_change()
sp_data = sp_data.dropna()
sp_data.set_index('Date',inplace = True)

# Random Walk
# simulate a random walk in one year
mu = np.mean(sp_data['Return'])
vol = np.std(sp_data['Return'])
T = 252
S0 = 10
rand_rets = np.random.normal(mu,vol,T) +1
forecasted_values = S0*rand_rets.cumprod()
# plt.plot(range(0,T), forecasted_values)
# plt.show()

# Monte Carlo Simulations
# a series of Monte Carlo simulations of a single asset starting at stock price S0 at T0
for i in range(100):
    rand_rets = np.random.normal(mu,vol,T)+1
    forecasted_values = S0 * rand_rets.cumprod()
    plt.plot(range(T),forecasted_values)
# plt.show()

# VaR(99)
# calculate the VaR(95) of 100 Monte Carlo simulations
sim_returns = []
for i in range(100):
    rand_ret = np.random.normal(mu,vol,T)
    sim_returns.append(rand_ret)
var_99 = np.percentile(sim_returns,1)
print("VaR(99): ",round(100*var_99,3),"%")
