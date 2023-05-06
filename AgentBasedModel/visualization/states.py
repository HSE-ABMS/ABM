from AgentBasedModel.simulator import SimulatorInfo
from AgentBasedModel.states import states_pattern
import AgentBasedModel.utils.math as math
import AgentBasedModel.utils.states_math as st_math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def plot_bear_trend_all_indicators(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Trends with all indicators')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_price = np.array(math.rolling(info.prices, rolling))
    plt.plot(range(rolling - 1, len(rolling_price)), rolling_price, color='black')

    bear_trend_adx = states_pattern.bear_trend_adx(rolling_price, window)
    bull_trend_adx = states_pattern.bull_trend_adx(rolling_price, window)
    plt.plot(bear_trend_adx, rolling_price[bear_trend_adx], '.', color='darkred')
    plt.plot(bull_trend_adx, rolling_price[bull_trend_adx], '.', color='darkolivegreen')

    bear_trend_extr, bull_trend_extr = states_pattern.trend_extremum(rolling_price, window)
    plt.plot(bear_trend_extr, np.array(rolling_price)[bear_trend_extr], '.', color='red')
    plt.plot(bull_trend_extr, np.array(rolling_price)[bull_trend_extr], '.', color='green')

    static_adf_result = states_pattern.adfuller_test(rolling_price, window)
    plt.plot(static_adf_result, rolling_price[static_adf_result], '.', color='pink')
    plt.show()

def plot_trends_local_extremums(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Trends with extremums')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_price = np.array(math.rolling(info.prices, rolling))
    plt.plot(range(rolling - 1, len(rolling_price)), rolling_price, color='black')

    bear_trend_extr, bull_trend_extr = states_pattern.trend_extremum(rolling_price, window)
    plt.plot(bear_trend_extr, np.array(rolling_price)[bear_trend_extr], '.', color='red')
    plt.plot(bull_trend_extr, np.array(rolling_price)[bull_trend_extr], '.', color='lime')
    plt.show()

def plot_trends_adx(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Trends with ADX')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_price = np.array(math.rolling(info.prices, rolling))
    plt.plot(range(rolling - 1, len(rolling_price)), rolling_price, color='black')

    bear_trend_adx = states_pattern.bear_trend_adx(rolling_price, window)
    bull_trend_adx = states_pattern.bull_trend_adx(rolling_price, window)
    plt.plot(bear_trend_adx, rolling_price[bear_trend_adx], '.', color='g')
    plt.plot(bull_trend_adx, rolling_price[bull_trend_adx], '.', color='r')
    plt.show()

def plot_trends_adfuller(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Trends with Adfuller test')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_price = np.array(math.rolling(info.prices, rolling))
    plt.plot(range(rolling - 1, len(rolling_price)), rolling_price, color='black')

    static_adf_result = states_pattern.adfuller_test(rolling_price, window)
    plt.plot(static_adf_result, rolling_price[static_adf_result], '.', color='pink')
    plt.show()

def plot_trends_linreg(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), size: int = 5, window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Trends with Linear Regression')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_prices = math.rolling(info.prices, rolling)
    indexes = np.arange(len(rolling_prices))
    for i in range(max(size,window), len(rolling_prices)-max(size, window), size):
        X = np.array(rolling_prices[i-window:window+i]).reshape(-1, 1)
        y = np.array(indexes[i-window:window+i]).reshape(-1, 1)
        result = states_pattern.linreg_trends(X, y)
        color = 'b'
        if result > 0.1:
            color = 'g'
        elif result < -0.1:
            color = 'r'
        plt.plot(indexes[i-size//2-1:i+size//2+1], rolling_prices[i-size//2 - 1:i+size//2 + 1], color=color)
    plt.show()

def plot_trends(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Trends')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_price = np.array(math.rolling(info.prices, rolling))
    plt.plot(range(rolling - 1, len(rolling_price)), rolling_price, color='black')

    bear_trend_adx = states_pattern.bear_trend_adx(rolling_price, window)
    bull_trend_adx = states_pattern.bull_trend_adx(rolling_price, window)

    bear_trend_extr, bull_trend_extr = states_pattern.trend_extremum(rolling_price, window)

    static_adf_result = states_pattern.adfuller_test(rolling_price, window)

    panic_indexes = states_pattern.panic_std_volatility(info.price_volatility(window))

    indexes = np.arange(len(rolling_price))
    for i in range(window, len(rolling_price) - window - 1, window):
        X = np.array(rolling_price[i-window:window+i]).reshape(-1, 1)
        y = np.array(indexes[i-window:window+i]).reshape(-1, 1)
        result = states_pattern.linreg_trends(X, y)
        color = 'grey'
        if result > 0.05:
            color = 'g'
        elif 0 < result < 0.05 and (i in bear_trend_adx or i in bear_trend_extr):
            color = 'g'
        elif result < -0.05:
            color = 'r'
        elif 0 > result > -0.05 and (i in bull_trend_adx or i in bull_trend_extr):
            color = 'r'
        elif i in static_adf_result:
            color = 'b'
        # if i in panic_indexes:
        #     color = 'orange'
        plt.plot(range(i, i + window + 1), rolling_price[i:window+i + 1], color=color)
    plt.show()

def plot_panic(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Panic')
    plt.xlabel('Iterations')
    plt.ylabel('Price')

    rolling_price = np.array(math.rolling(info.prices, rolling))
    plt.plot(range(rolling - 1, len(rolling_price)), rolling_price, color='black')

    panic_std_indexes = states_pattern.panic_std_volatility(np.array(info.price_volatility(window)))
    panic_ols_indexes = states_pattern.panic_ols_test(np.array(info.price_volatility(window)))
    panic_spreads = states_pattern.panic_spread(np.array(info.spreads))
    plt.plot(panic_std_indexes, rolling_price[panic_std_indexes], '.', color='r')
    plt.plot(panic_ols_indexes, rolling_price[panic_ols_indexes], '.', color='g')
    plt.plot(panic_spreads, rolling_price[panic_spreads], '.', color='y')
    plt.show()