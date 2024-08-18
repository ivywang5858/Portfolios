# Created by ivywang at 2024-08-09
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import scipy.optimize as sco
import scipy

# Setting the plotting style to be colorblind-friendly
plt.style.use("seaborn-v0_8-colorblind")

# Loading data
stock_prices_df = pd.read_csv("faang_stocks.csv", index_col="Date")

# Changing the index to a datetime type allows for easier filtering and plotting.
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
# stock_prices_df.plot(title="FAANG stock prices from years 2020-2023")
# plt.show()

# Task 1
# What are the expected returns and the annualized Sharpe ratio of an equally-weighted portfolio?
# Assume the risk-free rate is 0% and store your answers as a float variables called benchmark_exp_return and benchmark_sharpe_ratio.
rf = 0
weight = np.array([0.2]*5)
return_port_0 = stock_prices_df.pct_change().dropna()
cov_ma = return_port_0.cov()
cov = weight.T@cov_ma@weight
# print(np.sqrt(cov))

return_port = np.dot(return_port_0,weight)
return_port_mean = return_port.mean()
benchmark_exp_return = return_port_mean
benchmark_sharpe_ratio = (benchmark_exp_return-rf)/(return_port.std()*np.sqrt(252))
# print(return_port.std())
# print(benchmark_exp_return,benchmark_sharpe_ratio)

# Task 2
# Find a portfolio that minimizes volatility. Use mean-variance optimization.
# Store the volatility of the portfolio as a float variable called mv_portfolio_vol.
# Store the portfolio weights as a pandas Series called mv_portfolio. Use the tickers as index.

# Solution 2.1
# mu = expected_returns.mean_historical_return(stock_prices_df)
stock_return = stock_prices_df.pct_change().dropna()
mu = stock_return.mean()*252
cov = stock_prices_df.pct_change().dropna().cov()*252
ef = EfficientFrontier(mu,cov)
weights = ef.min_volatility()
# n = ef.clean_weights() # clean weights
mv_portfolio = pd.Series(weights)
# print(mv_portfolio)
mv_portfolio_vol = ef.portfolio_performance(risk_free_rate=rf)[1]
# print(mv_portfolio_vol)

# Solution 2.2
n = len(mu)
def get_port_vol(wgt,cov):
    return np.sqrt(np.dot(wgt.T,np.dot(cov,wgt)))
x0 = n*[1/n]
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
bnds = tuple((0,1) for x in range(n))
res = sco.minimize(get_port_vol, x0,cov, method='SLSQP', bounds=bnds, constraints=cons)
mv_portfolio = pd.Series(res.x, index=mu.index)
mv_portfolio_vol = res.fun
# print(mv_portfolio,mv_portfolio_vol)

# Task 3
# Find a portfolio that maximizes the Sharpe ratio. Use mean-variance optimization and keep the risk-free rate at 0%.
# Store the Sharpe ratio (annualized) of the portfolio as a float variable called ms_portfolio_sharpe.
# Store the portfolio weights as a pandas Series called ms_portfolio. Use the tickers as index.

# Solution 3.1
# Alternative approach to get the expected returns and the covariance matrix
avg_returns = expected_returns.mean_historical_return(stock_prices_df, compounding=False)
cov_mat = risk_models.sample_cov(stock_prices_df)

# Instantiate the EfficientFrontier object
ef = EfficientFrontier(avg_returns, cov_mat)

# Find the weights that maximize the Sharpe ratio
weights = ef.max_sharpe(risk_free_rate=0)
ms_portfolio = pd.Series(weights)

# Find the maximized Sharpe ratio
ms_portfolio_sharpe = ef.portfolio_performance(risk_free_rate=0)[2]
# print(ms_portfolio)
# print(ms_portfolio_sharpe)

# Solution 3.2
n = len(mu)
def get_sharpe_ratio(wgt,cov,stock_return):
    return 1/((np.dot(mu,wgt))/np.sqrt(np.dot(wgt.T,np.dot(cov,wgt))))
x0 = n*[1/n]
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
bnds = tuple((0,1) for i in range(n))
res = sco.minimize(get_sharpe_ratio, x0,args=(cov,stock_return), method='SLSQP', bounds=bnds, constraints=cons)
mv_portfolio = pd.Series(res.x, index=mu.index)
ms_portfolio_sharpe =  1/res.fun
# print(mv_portfolio)
# print(ms_portfolio_sharpe)

# https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb
# https://github.com/omartinsky/BlackLitterman/blob/master/black_litterman.ipynb (scipy.optimize)
# https://www.quantandfinancial.com/2013/07/mean-variance-portfolio-optimization.html
# https://medium.com/data-driven-investment/%E8%B3%87%E7%94%A2%E9%85%8D%E7%BD%AE-%E7%AC%AC%E4%B8%89%E4%BB%A3%E8%B3%87%E7%94%A2%E9%85%8D%E7%BD%AE%E7%90%86%E8%AB%96-black-litterman-b4d2fd855dad


# Max Drawdown
# Solution 1 - price drawdown
# Calculate the max value
stock_prices_df = stock_prices_df['AAPL']
roll_max = stock_prices_df.rolling(center=False,min_periods=1,window=252).max()
# Calculate the daily draw-down relative to the max
daily_draw_down = stock_prices_df/roll_max - 1.0
# Calculate the minimum (negative) daily draw-down
max_daily_draw_down = daily_draw_down.rolling(center=False,min_periods=1,window=252).min()
# # Plot the results
# plot1 = plt.subplot2grid((1, 2), (0, 0))
# plot2 = plt.subplot2grid((1, 2), (0, 1))
# # plt.figure(figsize=(15,15))
# plot1.plot(stock_prices_df.index, daily_draw_down, label='Daily drawdown')
# plot1.plot(stock_prices_df.index, max_daily_draw_down, label='Maximum daily drawdown in time-window')
# plt.legend()
# # plt.show()

# Solution 2 - return drawdown
stock_return_df = stock_prices_df.pct_change().dropna()
# Calculate the running maximum
running_max = np.maximum.accumulate(stock_return_df)
# Ensure the value never drops below 1
# running_max[running_max < 1] = 1
# Calculate the percentage drawdown
drawdown = (stock_return_df)/running_max - 1
# Plot the results
# plot2.plot(running_max.index,drawdown,label='Daily drawdown')
# plt.show()

# Sortino ratio
stock_return_df = stock_prices_df.pct_change().dropna()
# Create a downside return column with the negative returns only
downside_returns = stock_return_df.loc[stock_return_df < stock_return_df.mean()]
# Calculate expected return and std dev of downside
expected_return = stock_return_df.mean()
down_stdev = downside_returns.std()
# Calculate the sortino ratio
sortino_ratio = (expected_return - 0)/down_stdev
# Print the results
# print("Expected return  : ", expected_return*100)
# print("Downside risk   : ", down_stdev*100)
# print("Sortino ratio : ", sortino_ratio)

# VaR
# Calculate historical VaR(95)
var_95 = np.percentile(stock_return_df, 5)
# print(var_95)
# Sort the returns for plotting
sorted_rets = sorted(stock_return_df)
# Plot the probability of each sorted return quantile
plt.hist(sorted_rets, density=True, stacked=True)
# Denote the VaR 95 quantile
plt.axvline(x=var_95, color='r', linestyle='-', label='VaR 95: {0:.2f}%'.format(var_95))
# plt.show()

#CVaR
# Historical CVaR 95
cvar_95 = stock_return_df[stock_return_df <= var_95].mean()
# print(cvar_95)
# Sort the returns for plotting
sorted_rets = sorted(stock_return_df)
# Plot the probability of each return quantile
plt.hist(sorted_rets, density=True, stacked=True)
# Denote the VaR 95 and CVaR 95 quantiles
plt.axvline(x=var_95, color="r", linestyle="-", label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(x=cvar_95, color='b', linestyle='-', label='CVaR 95: {0:.2f}%'.format(cvar_95))
# plt.show()

losses = pd.Series(scipy.stats.norm.rvs(size = 1000))
VaR_95 = scipy.stats.norm.ppf(0.95)
CVaR_95 = (1/(1-0.95))*scipy.stats.norm.expect(lambda x: x,lb = VaR_95)
print(CVaR_95)

# Parametric VaR
# Import norm from scipy.stats
from scipy.stats import norm
# Estimate the average daily return
mu = np.mean(stock_return_df)
# Estimate the daily volatility
vol = np.std(stock_return_df)
# Set the VaR confidence level
confidence_level = 0.05
# Calculate Parametric VaR
var_95 = norm.ppf(confidence_level, mu, vol)
# print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95))
