# Created by ivywang at 2024-02-05
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GARCH_model import return_processing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#Augmented Dicky-Fuller test - test for stationarity
def ADF_test(data):
    data = data.drop(columns=['Return'])
    result = adfuller(data['Price'])
    print("Test statistic: ",result[0])
    print("p-value: ",result[1])
    print("critical value: ",result[4])
    fig, ax = plt.subplots()
    data.plot(ax = ax)
    plt.show()

    # calculate the first difference and run Augmented Dicky-Fuller test
    aapl_diff = data.diff().dropna()
    result_diff = adfuller(aapl_diff['Price'])
    print(result_diff)

# Generate ARMA data
def Gen_ARMA_data():
    np.random.seed(1)
    ar_coefs = [1]
    ma_coefs = [1,-0.7]
    # generate ARMA data
    y = arma_generate_sample(ar_coefs,ma_coefs,nsample = 100, scale = 0.5)
    plt.plot(y)
    plt.ylabel(r'$y_t$')
    plt.xlabel(r'$t$')
    plt.show()

# Generate one-step-ahead predictions
def one_step_ahead_pred(data):
    # istantiate the model ARMA
    data = data.loc['2023-09-01':]
    model = ARIMA(data['Price'],order = (1,0,1))
    results = model.fit()
    # print(results.summary())

    # generate predictions
    one_step_forecast = results.get_prediction(start = -30)
    # extract prediction mean
    mean_forecast = one_step_forecast.predicted_mean
    # get confidence intervals of predictions
    conf_int = one_step_forecast.conf_int()
    lower_limits = conf_int.loc[:,'lower Price']
    upper_limits = conf_int.loc[:,'upper Price']
    # print(mean_forecast)

    # plot the apple data
    plt.plot(data.index,data['Price'],label = 'observed')
    # plot the mean predictions
    plt.plot(mean_forecast.index,mean_forecast,color = 'r', label = 'forecast')
    # shade the area of confidence interval
    plt.fill_between(mean_forecast.index, lower_limits, upper_limits, color = 'pink')
    plt.xlabel('Date')
    plt.ylabel('AAPL Stock Price')
    plt.legend()
    plt.show()

# Generate dynamic forecasts
def dynamic_forecast(data):
    # istantiate the model ARMA
    data = data.loc['2023-09-01':]
    model = ARIMA(data['Price'], order=(1, 0, 1))
    results = model.fit()

    # generate predictions
    dynamic_forecast = results.get_prediction(start= -30, dynamic = True)
    # extract prediction mean
    mean_forecast = dynamic_forecast.predicted_mean
    # get confidence intervals of predictions
    conf_int = dynamic_forecast.conf_int()
    lower_limits = conf_int.loc[:, 'lower Price']
    upper_limits = conf_int.loc[:, 'upper Price']

    # plot the apple data
    plt.plot(data.index, data['Price'], label='observed')
    # plot the mean predictions
    plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
    # shade the area of confidence interval
    plt.fill_between(mean_forecast.index, lower_limits, upper_limits, color='pink')
    plt.xlabel('Date')
    plt.ylabel('AAPL Stock Price')
    plt.legend()
    plt.show()

# Differencing and fitting ARMA
def diff_fit_ARMA(data):
    data.index = pd.DatetimeIndex(data.index).to_period('D')
    # take the first difference
    data_diff = data['Price'].diff().dropna()
    # create ARMA(2,2) model
    arma = ARIMA(data_diff,order = (2,0,2))
    # Fit model
    arma_results = arma.fit()
    # print(arma_results.summary())

    # Unrolling ARMA forecast - forecast the abs value of price dataset
    # Make arma forecast of next 10 differences
    arma_diff_forecast = arma_results.get_forecast(steps = 10).predicted_mean
    # print(arma_diff_forecast)
    # Integrate the diff forecast
    arma_int_forecast = np.cumsum(arma_diff_forecast)
    # Make abs value forecast
    arma_value_forecast = arma_int_forecast + data.iloc[-1,0]
    print(arma_value_forecast)

# Fit using ARIMA model
def fit_ARIMA(data):
    # Create ARIMA(2,1,2) model
    arima = ARIMA(data['Price'], order = (2,1,2))
    # Fit ARIMA model
    arima_results = arima.fit()
    # Make ARIMA forecast of next 10 values
    arima_value_forecast = arima_results.get_forecast(steps = 10).predicted_mean
    print(arima_value_forecast)

def ACF_PACF(data):
    # AR(p): ACF - tails off, PACF - cuts off after lag p
    # MA(q): ACF - cuts off after lag p, PACF - tails off
    # ARMA(p,q): ACF - tails off,  PACF - tails off
    data = data['Price']
    # create figure
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,8))
    # ACF
    plot_acf(data,lags=10,zero=False,ax = ax1)
    # PACF
    plot_pacf(data,lags=10,zero=False,ax = ax2)
    plt.show()



def main():
    # AAPL
    AAPL_data = return_processing('AAPL 2020.csv')

    #ARIMA Model functions
    # ADF_test(AAPL_data)
    # Gen_ARMA_data()
    # one_step_ahead_pred(AAPL_data)
    # dynamic_forecast(AAPL_data)
    # diff_fit_ARMA(AAPL_data)
    # fit_ARIMA(AAPL_data)

    # The best of the best models
    ACF_PACF(AAPL_data)




'''Main Function'''
if __name__ == '__main__':
    main()