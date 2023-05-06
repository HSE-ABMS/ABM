from AgentBasedModel.simulator import SimulatorInfo
from AgentBasedModel.states import states_pattern
import AgentBasedModel.utils.math as math
import AgentBasedModel.utils.states_math as st_math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def plot_moving_average(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Moving Average')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    rolling_price = math.rolling(info.prices, rolling)
    moving_average = st_math.moving_average(rolling_price, window)
    plt.plot(range(rolling - 1, len(info.prices)), rolling_price, color='black')
    plt.plot(range(window, len(moving_average) + window), moving_average, color='red')
    plt.show()

def plot_exp_moving_average(info: SimulatorInfo, rolling: int = 1, figsize=(12, 6), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Exponential Moving Average')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    rolling_price = math.rolling(info.prices, rolling)
    exp_moving_average = st_math.exp_moving_average(rolling_price, window)
    plt.plot(range(rolling - 1, len(info.prices)), rolling_price, color='black')
    plt.plot(range(rolling - 1, len(exp_moving_average)), color='red')
    plt.show()

def plot_adx(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Average Directional Index')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    di_plus, di_minus, adx = st_math.average_directional_index(math.rolling(info.prices, rolling), window)
    plt.plot(range(0,len(adx)*5, 5), adx, color='b')
    plt.plot(range(0,len(adx)*5, 5), [20] * len(adx), color='y')
    plt.plot(range(rolling - 1, len(di_minus) * 5, 5), di_minus, color='g')
    plt.plot(range(rolling - 1, len(di_plus) * 5, 5), di_plus, color='r')
    plt.legend(title='Indicators', labels=['ADX', '20-th level', '-DI', '+DI'])
    plt.show()

def plot_adfuller(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7), size: int = 25, window: int = 5):
    plt.figure(figsize=figsize)
    plt.title('Adfuller Test')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    prices = np.array(math.rolling(info.prices, rolling))
    indexes = np.arange(len(prices))
    adfuller_indexes = states_pattern.adfuller_test(prices)
    plt.plot(range(len(prices)), prices, color='black')
    plt.plot(adfuller_indexes, prices[adfuller_indexes], '.', color='red')
    plt.show()

def plot_standart_moving_average(info: SimulatorInfo, rolling: int = 1, figsize=(14, 7)):
    plt.figure(figsize=figsize)
    plt.title('Moving Average')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    prices = math.rolling(info.prices, rolling)
    plt.plot(range(rolling - 1, len(prices)), prices, color='black')
    moving_average_25 = st_math.moving_average(prices, 25)
    moving_average_100 = st_math.moving_average(prices, 100)
    moving_average_200 = st_math.moving_average(prices, 200)
    plt.plot(range(25, len(moving_average_25) + 25), moving_average_25, color='red')
    plt.plot(range(100, len(moving_average_100) + 100), moving_average_100, color='green')
    plt.plot(range(200, len(moving_average_200) + 200), moving_average_200, color='blue')
    plt.legend(title='MA', labels=['prices', 'window=25', 'window=100', 'window=200'])
    plt.show()
