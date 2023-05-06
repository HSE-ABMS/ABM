import numpy as np

def moving_average(array: list, window: int = 15):
    cumsum = np.cumsum(array)
    return (cumsum[window:] - cumsum[:-window]) / window

def exp_moving_average(array: list, window: int = 3, x = 0.5):
    moving_averages = [array[0]]
    for i in range(1, len(array)):
        window_average = round((x*array[i])+(1-x)*moving_averages[-1], 2)
        moving_averages.append(window_average)
    return np.array(moving_averages)
        
def average_true_range(array: list, window: int = 15):
    true_range = np.zeros(len(array) // window)
    for i in range(window, len(array)-window, window):
        true_range[i // window] = max(max(array[i:i+window]) - min(array[i:i+window]),
                                      max(array[i:i+window]) - array[i - 1],
                                      array[i - 1] - min(array[i:i+window]))
    return moving_average(true_range)

def average_directional_index(array: list, window: int = 5):
    dm_plus = np.zeros(len(array) // window)
    dm_minus = np.zeros(len(array) // window)
    for i in range(window, len(array) - window, window):
        up_move = max(array[i:i+window]) - max(array[i-window:i])
        down_move = min(array[i-window:i]) - min(array[i:i+window])
        if up_move > down_move and up_move > 0:
            dm_plus[i // window] = up_move
        if up_move < down_move and down_move > 0:
            dm_minus[i // window] = down_move
    average_true = average_true_range(array, window=window)
    di_plus = 100 * moving_average(dm_plus[1 + 15:] / average_true[1:])
    di_minus = 100 * moving_average(dm_minus[1 + 15:] / average_true[1:])
    adx = 100 * moving_average(abs(di_plus - di_minus) / (di_plus + di_minus + 0.01))
    return di_plus, di_minus, adx


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
