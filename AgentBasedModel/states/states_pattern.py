from AgentBasedModel.simulator import Simulator, SimulatorInfo

import numpy as np

def moving_average(prices: list, size: int = 25):
    cumsum = np.cumsum(prices)
    return (cumsum[size:] - cumsum[:-size]) / size

def exp_moving_average(prices: list, size: int = 25):
    cumsum = np.exp(prices)
    return (cumsum[size:] - cumsum[:-size]) / size

# def order_book_analyze():


def bear_trend(info: SimulatorInfo, size: int = None, window: int = 5) -> bool or list:
    mov_avg = moving_average(info.prices, window)
    result = np.argwhere(mov_avg > info.prices)
    return result


