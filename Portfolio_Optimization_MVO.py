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
return_port = stock_prices_df.pct_change().dropna()
cov_ma = return_port.cov()
cov = weight.T@cov_ma@weight
print(np.sqrt(cov))

return_port = np.dot(return_port,weight)
return_port_mean = return_port.mean()
benchmark_exp_return = return_port_mean
benchmark_sharpe_ratio = (benchmark_exp_return-rf)/(return_port.std()*np.sqrt(252))
print(return_port.std())
print(benchmark_exp_return,benchmark_sharpe_ratio)



# Task 2
# Find a portfolio that minimizes volatility. Use mean-variance optimization.
# Store the volatility of the portfolio as a float variable called mv_portfolio_vol.
# Store the portfolio weights as a pandas Series called mv_portfolio. Use the tickers as index.



# Task 3
# Find a portfolio that maximizes the Sharpe ratio. Use mean-variance optimization and keep the risk-free rate at 0%.
# Store the Sharpe ratio (annualized) of the portfolio as a float variable called ms_portfolio_sharpe.
# Store the portfolio weights as a pandas Series called ms_portfolio. Use the tickers as index.