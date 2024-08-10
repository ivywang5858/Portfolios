# Created by ivywang at 2024-08-09
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import scipy.optimize as sco

# Setting the plotting style to be colorblind-friendly
plt.style.use("seaborn-v0_8-colorblind")

# Loading data
stock_prices_df = pd.read_csv("faang_stocks.csv", index_col="Date")

# Changing the index to a datetime type allows for easier filtering and plotting.
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
stock_prices_df.plot(title="FAANG stock prices from years 2020-2023")
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
print(ms_portfolio)
print(ms_portfolio_sharpe)


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
print(mv_portfolio)
print(ms_portfolio_sharpe)



# https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb
# https://github.com/omartinsky/BlackLitterman/blob/master/black_litterman.ipynb (scipy.optimize)
# https://www.quantandfinancial.com/2013/07/mean-variance-portfolio-optimization.html
# https://medium.com/data-driven-investment/%E8%B3%87%E7%94%A2%E9%85%8D%E7%BD%AE-%E7%AC%AC%E4%B8%89%E4%BB%A3%E8%B3%87%E7%94%A2%E9%85%8D%E7%BD%AE%E7%90%86%E8%AB%96-black-litterman-b4d2fd855dad