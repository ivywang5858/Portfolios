# Created by ivywang at 2024-01-14
import pandas as pd
import numpy as np
from os import chdir
from arch import arch_model
import matplotlib.pyplot as plt


def return_processing(file):
    data = pd.read_csv(file)
    data = data[['Date', 'Adj Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.rename(columns={'Adj Close': 'Price'})
    data['Return'] = data['Price'].pct_change() * 100
    data = data.dropna()
    data.set_index('Date', inplace=True)
    return data
def GARCH_model_analysis(data):
    # GARCH Model Fit
    basic_gm = arch_model(data['Return'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
    gm_result = basic_gm.fit(update_freq = 4,disp='off')
    print(gm_result.summary())
    gm_result.plot()
    plt.show()

    # GARCH Model Forecast
    gm_result = basic_gm.fit(disp= 'off')
    gm_forecast = gm_result.forecast(horizon = 5)
    print(gm_forecast.variance[-1:])

def mean_model(data):
    # Mean Model Specifications - mean uses AR model
    # constant mean
    constant_mean_result = arch_model(data['Return'],p =1, q = 1, mean = 'constant', vol = 'GARCH')
    constant_mean_model = constant_mean_result.fit(disp= 'off')
    print(constant_mean_model.summary())
    cmean_vol = constant_mean_model.conditional_volatility

    # autoregression mean
    ar_mean_result = arch_model(data['Return'],p =1, q = 1, mean = 'AR', vol = 'GARCH')
    ar_mean_model = ar_mean_result.fit(disp= 'off')
    print(ar_mean_model.summary())
    armean_vol = ar_mean_model.conditional_volatility

    plt.plot(cmean_vol,color = 'blue', label = 'Constant Mean Volatility')
    plt.plot(armean_vol,color = 'red', label = 'AR Mean Volatility')
    plt.legend(loc = 'upper right')
    plt.show()
    print(np.corrcoef(cmean_vol,armean_vol)[0,1])

def vol_model_asym(data):
    # Volatility Models for asymmetric shocks - GJR GARCH & EGARCH
    # GJR GARCH: o = 1, vol = 'GARCH'
    gjr_gm = arch_model(data['Return'],p =1, q = 1, o = 1, vol = 'GARCH', dist = 't' )
    gjr_result = gjr_gm.fit(disp = 'off')
    print(gjr_result.summary())
    gjrgm_vol = gjr_result.conditional_volatility

    # EGARCH: o = 1, vol = 'EGARCH'
    egarch_gm = arch_model(data['Return'],p =1, q = 1, o = 1, vol = 'EGARCH', dist = 't' )
    egarch_result = egarch_gm.fit(disp = 'off')
    print(egarch_result.summary())
    egarch_vol = egarch_result.conditional_volatility

    # plots of price returns, GJR GARCH vol and EGARCH vol (forecasted)
    plt.plot(data['Return'], color = 'grey',alpha = 0.4, label = 'Price Returns')
    plt.plot(gjrgm_vol, color = 'gold', label = 'GJR-GARCH Volatility')
    plt.plot(egarch_vol, color = 'red', label = 'EGARCH Volatility')
    plt.legend(loc = 'upper right')
    plt.show()

def rolling_window_forecast(data):
    # GARCH Rolling Window Forecast
    first_obs = 0
    end_obs = 100
    forecasts_fi= {}
    forecasts_ex= {}
    ob_var = {}
    basic_gm = arch_model(data['Return'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
    # basic_gm2 = arch_model(sp_data['Return'][:'2021-08-03'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'normal')
    # gm_result2 = basic_gm2.fit(update_freq=5, disp='off')
    # temp_result2 = gm_result2.forecast(horizon=1).variance
    # print(temp_result2)

    # Rolling Window to forecast variance
    # Fixed Window
    for i in range(300):
        gm_result = basic_gm.fit(first_obs= i+first_obs, last_obs= i+end_obs, update_freq = 5, disp = 'off')
        temp_result = gm_result.forecast(horizon = 1).variance
        fcast = temp_result.iloc[0]
        forecasts_fi[fcast.name] = fcast
    forecast_fi_var = pd.DataFrame(forecasts_fi).T

    # Expanding Window
    for i in range(300):
        gm_result_ex = basic_gm.fit(first_obs= first_obs, last_obs= i+end_obs, update_freq = 5, disp = 'off')
        temp_result_ex = gm_result_ex.forecast(horizon = 1).variance
        excast = temp_result_ex.iloc[0]
        forecasts_ex[excast.name] = excast

        # #real vol
        # real_var = sp_data['Return'][i+first_obs:i+end_obs].var()
        # ob_var[excast.name] = real_var
    forecast_ex_var = pd.DataFrame(forecasts_ex).T
    # var_pd = pd.DataFrame(ob_var,index=[0]).T

    # Plot the forecast variance and price return
    plt.subplot(2,1,1)
    plt.plot(forecast_fi_var, color = 'red',label = 'Fixed Window Forecasted Variance')
    plt.plot(data.Return['2020-05-27':'2021-08-03'],color='gold',label = 'Price Return')
    plt.legend(loc = 'upper right')
    # plt.plot(var_pd['2020-05-27':'2021-08-03'],color='blue')


    # Plot both volatility forecast - expending and fixed window and price return
    vol_fixedwin = np.sqrt(forecast_fi_var)
    vol_expandwin = np.sqrt(forecast_ex_var)
    plt.subplot(2,1,2)
    plt.plot(vol_fixedwin, color = 'red',label = 'Fixed Window Forecasted Volatility')
    plt.plot(vol_expandwin, color = 'blue',label = 'Expanding Window Forecasted Volatility')
    plt.plot(data['Return']['2020-05-27':'2021-08-04'],color='gold',label = 'Price Return')
    plt.legend(loc = 'lower right')
    plt.show()

def VaR_analysis(data):
    # Value at Risk VaR with GARCH Model
    mean_f = {}
    basic_gm = arch_model(data['Return'], p =1, q = 1, mean = 'constant', vol = 'GARCH', dist = 't')
    gm_result = basic_gm.fit(disp='off')
    gm_forecast = gm_result.forecast(start = '2022-12-31')
    # obtain forword-looking mean and volatility
    mean_forecast = gm_forecast.mean['2022-12-31':]
    variance_forecast = gm_forecast.variance['2022-12-31':]

    # obtain the parametric quantile (model parameters)
    nu = gm_result.params.iloc[4]
    q_parametric = basic_gm.distribution.ppf(0.05,nu)
    print('5% parametric quantile: ',q_parametric)
    # calculate the VaR
    VaR_parametric = mean_forecast.values + np.sqrt(variance_forecast).values*q_parametric
    VaR_parametric = pd.DataFrame(VaR_parametric,columns=['5%'],index = variance_forecast.index)

    # plot the VaR
    plt.subplot(2,1,1)
    plt.plot(VaR_parametric,color = 'red',label = '5% Parametric VaR')
    plt.scatter(variance_forecast.index,data['Return']['2022-12-31':], color = 'orange', label = 'Daily Returns')
    plt.legend(loc = 'upper right')

    # obtain the empirical quantile (observations/ data)
    std_resid_emp = gm_result.resid/gm_result.conditional_volatility
    q_empirical = std_resid_emp.quantile(0.05)
    print('5% empirical quantile: ',q_empirical)

    # calculate the VaR
    VaR_empirical = mean_forecast.values + np.sqrt(variance_forecast).values*q_empirical
    VaR_empirical = pd.DataFrame(VaR_empirical,columns=['5%'],index = variance_forecast.index)

    # plot the VaRs
    plt.subplot(2,1,2)
    plt.plot(VaR_empirical, color = 'blue', label='5% Expirical VaR')
    plt.plot(VaR_parametric, color = 'red', label='5% Parametric VaR')
    plt.scatter(variance_forecast.index,data['Return']['2022-12-31':], color = 'orange', label = 'Daily Returns')
    plt.legend(loc='upper right')
    plt.show()

def dynamic_covariance(data_a, data_b):
    # Dynamic Covariance in portfolio optimization
    basic_gm_sp = arch_model(data_a['Return']['2020-01-02':'2020-12-31'], p=1, q=1, mean='constant', vol='GARCH', dist='t')
    gm_result_sp = basic_gm_sp.fit(disp='off')
    basic_gm_bitcoin = arch_model(data_b['Return']['2020-01-02':'2020-12-31'], p=1, q=1, mean='constant', vol='GARCH',dist='t')
    gm_result_bitcoin = basic_gm_bitcoin.fit(disp='off')
    sp_vol = gm_result_sp.conditional_volatility
    sp_std_resid = gm_result_sp.resid/sp_vol
    bitcoin_vol = gm_result_bitcoin.conditional_volatility
    bitcoin_std_resid = gm_result_bitcoin.resid/bitcoin_vol

    # calc correlation
    corr = np.corrcoef(sp_std_resid,bitcoin_std_resid)[0,1]
    print('Correlation: ', corr)




def main():
    # S&P 500 return cleaning
    sp_data = return_processing('S&P500 2020.csv')
    # Bitcoin return cleaning
    bitcoin_data = return_processing('Bitcoin 2020.csv')
    # AAPL
    AAPL_data = return_processing('AAPL 2020.csv')
    # Fidelity U.S. Bond Index
    FXNAX_data = return_processing('FXNAX 2020.csv')



    # ---GARCH Model functions---
    # GARCH_model_analysis(sp_data)
    # mean_model(sp_data)
    # vol_model_asym(bitcoin_data)
    # rolling_window_forecast(sp_data)
    # VaR_analysis(bitcoin_data)
    dynamic_covariance(sp_data,AAPL_data)


'''Main Function'''
if __name__ == '__main__':
    main()