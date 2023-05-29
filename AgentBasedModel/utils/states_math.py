import AgentBasedModel.utils.math as math

import numpy as np
from sklearn.linear_model import LinearRegression

def rolling(array: np.array, n) -> np.array:
    return np.array(math.rolling(array, n))

def moving_average(array: np.array, size: int = 10, window: int = 5) -> np.array:
    cumsum = np.cumsum(array)
    return (cumsum[window:] - cumsum[:-window]) / window
        
def average_true_range(array: np.array, size: int = 10, window: int = 5) -> np.array:
    true_range = np.zeros(len(array) // size)
    for i in range(1, len(array) // size):
        true_range[i] = max(max(array[i * size - window: i * size + window]) - min(array[i * size - window: i * size + window]),
                                      max(array[i * size - window: i * size + window]) - array[i * size - window],
                                      array[i * size - window] - min(array[i * size - window: i * size + window]))
    return moving_average(true_range)

def average_directional_index(array: np.array, size: int = 10, window: int = 5) -> np.array:
    dm_plus = np.zeros(len(array) // size)
    dm_minus = np.zeros(len(array) // size)
    for i in range(1, len(array) // size):
        up_move = max(array[i * size - window: i * size]) - max(array[i * size: i * size + window])
        down_move = min(array[i * size - window: i * size]) - min(array[i * size - window: i * size ])
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if up_move < down_move and down_move > 0:
            dm_minus[i] = down_move
    average_true = average_true_range(array, size=size, window=window)
    di_plus = 100 * moving_average(dm_plus[window:] / average_true)
    di_minus = 100 * moving_average(dm_minus[window:] / average_true)
    adx = 100 * moving_average(abs(di_plus - di_minus) / (di_plus + di_minus + 0.01))
    return di_plus[window:], di_minus[window:], adx


def adfuller_check(result: dict, p_value, window: int = 5) -> bool:
    if p_value < 0.05:
        return True
    if window == 25:
        return result['1%'] >= -3.75 and result['4%'] >= -3.0
    if window == 50:
        return result['1%'] >= -3.58 and result['4%'] >= -2.93
    if window == 100:
        return result['1%'] >= -3.51 and result['4%'] >= -2.89
    if window == 250:
        return result['1%'] >= -3.46 and result['4%'] >= -2.88
    if window == 500:
        return result['1%'] >= -3.44 and result['4%'] >= -2.87
    return False

def linear_regression(X: np.array, y: np.array) -> int:
    regr = LinearRegression()
    regr.fit(X, y)
    return regr.coef_[0][0]
