import AgentBasedModel.utils.states_math as st_math
from AgentBasedModel.states.states import test_trend_ols
import AgentBasedModel.utils.math as math

import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

def bear_trend_adx(prices: np.array, window: int = 15) -> list:
    mov_avg = st_math.moving_average(prices, window)
    mov_avg = np.argwhere(mov_avg > prices[:-window])
    di_plus, di_minus, adx_ = st_math.average_directional_index(prices)
    indexes = np.intersect1d(np.argwhere(di_plus > di_minus), np.argwhere(adx_ > 20))
    return indexes * window
    
def bull_trend_adx(prices: np.array, window: int = 5) -> list:
    mov_avg = st_math.moving_average(prices, window)
    mov_avg = np.argwhere(mov_avg < prices[:-window])
    di_plus, di_minus, adx_ = st_math.average_directional_index(prices)
    indexes = np.intersect1d(np.argwhere(di_plus < di_minus), np.argwhere(adx_ > 20))
    return indexes * window

def trend_extremum(prices: np.array, window: int = 15) -> list:
    bear_result = []
    bull_result = []
    for i in range(window, len(prices)-window, window):
        if prices[i] >= 1.15 * min(prices[max(0, i - len(prices)//window):i + window]):
            bear_result.append(i)
        if prices[i] * 1.15 <= max(prices[max(0, i - len(prices)//window):i+window]):
            bull_result.append(i)
    return bear_result, bull_result

def adfuller_test(prices: np.array, window: int = 5) -> list:
    adfuller_result = []
    for i in range(window * 10, len(prices)- window * 10, window):
        result = adfuller(prices[i-window * 10:window * 10+i])
        if st_math.adfuller_check(result[4], result[1]):
            adfuller_result.append(i)
    return np.array(adfuller_result)

def linreg_trends(X: np.array, y: np.array) -> int:
    regr = LinearRegression()
    regr.fit(X, y)
    return regr.coef_[0][0]

def panic_std_volatility(volatilities: np.array, window: int = 5, tol: float = 5) -> list:
    vol_result = []
    for i in range(window, len(volatilities) - window, window):
        if max(volatilities[i-window:i+window]) >= tol * math.mean(volatilities):
            vol_result.append(i)
    return vol_result

def panic_ols_test(vols: np.array, window: int = 5, tol: float = 0.2, conf = 0.95) -> list:
    size = window * 2
    vol_result = []
    for i in range(size, len(vols) - size, size):
        test = test_trend_ols(vols[i-size:i+size])
        if test['value'] > tol and test['p-value'] < (1 - conf):
            vol_result.append(i)
    return vol_result

def panic_extremum(volatilities: np.array, window: int = 15) -> list:
    result = []
    for i in range(window, len(volatilities)-window, window):
        if volatilities[i] * 1.5 <= max(volatilities[0:i+window]):
            result.append(i)
    return result

# def panic_order_book(quantities: np.array, window: int = 15, tol = 0.2) -> list:
#     result = []
#     print(quantities[0:2]['quantities'])
#     for i in range(window, len(quantities)-window, window):
#         ask_count = 0
#         bid_count = 0
#         for i in range(i-window, i+window):
#             ask_count += quantities[i]['quantities']['ask']
#             bid_count += quantities[i]['quantities']['bid']
#         if :
#             result.append(i)
#     return result

def panic_spread(spreads: np.array, window: int = 15) -> list:
    result = []
    mean_spread = 0
    for i in range(len(spreads)):
        mean_spread += spreads[i]['ask'] - spreads[i]['bid']
    mean_spread /= len(spreads)
    for i in range(window, len(spreads)-window, window):
        max_spread = 0
        for i in range(i-window, i+window):
            max_spread = max(max_spread, spreads[i]['ask'] - spreads[i]['bid'])
        if (spreads[i]['ask'] - spreads[i]['bid']) >= mean_spread:
            result.append(i)
    return result