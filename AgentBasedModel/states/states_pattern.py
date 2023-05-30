import AgentBasedModel.utils.states_math as st_math
from AgentBasedModel.simulator import SimulatorInfo

import numpy as np
np.seterr(divide = 'ignore') 
from statsmodels.tsa.stattools import adfuller

class StateIdentification:
    def __init__(self, info: SimulatorInfo = None, rolling: int = 1, size: int = 10, window: int = 5):
        if info is not None:
            self.prices = st_math.rolling(info.prices, rolling)
            self.volatility = info.price_volatility(window)
        self.rolling = rolling
        self.window = window
        self.size = size

    def set_params(self, adx_th: int = 20, max_th: float = 1.2, min_th: float = 1.2, linreg_th: float = 0.1, panic_std_th: float = 2.5, panic_extr_th: float = 1.5):
        self.adx_th = adx_th
        self.max_th = max_th
        self.min_th = min_th
        self.linreg_th = linreg_th
        self.panic_std_th = panic_std_th
        self.panic_extr_th = panic_extr_th
    
    def bear_adx(self) -> list:
        di_plus, di_minus, adx = st_math.average_directional_index(self.prices)
        indexes = np.intersect1d(np.argwhere(di_plus > di_minus), np.argwhere(adx > self.adx_th))
        return indexes + (self.window + self.size)
    
    def bull_adx(self) -> list:
        di_plus, di_minus, adx = st_math.average_directional_index(self.prices)
        indexes = np.intersect1d(np.argwhere(di_plus < di_minus), np.argwhere(adx > self.adx_th))
        return indexes + (self.window + self.size)
    
    def extremum(self) -> list:
        bear_result = []
        bull_result = []
        for i in range(self.size, len(self.prices), self.size):
            if max(self.prices[i - self.window:i+self.window]) >= self.max_th * min(self.prices[max(0, i + self.window - len(self.prices)//self.window):i +self.size + self.window]):
                bear_result.append(i)
            if min(self.prices[i - self.window:i+self.window]) < self.min_th * max(self.prices[max(0, i + self.window - len(self.prices)//self.window):i +self.size+ self.window]):
                bull_result.append(i)
        return bear_result, bull_result
    
    def linear_regression(self) -> np.array:
        coefs = []
        for i in range(self.size, len(self.prices) - self.window - 1, self.size):
            X = self.prices[i -self.window:min(len(self.prices), (i + self.size + self.window))].reshape(-1, 1)
            y = np.arange(i -self.window, min(len(self.prices), i + self.size + self.window)).reshape(-1, 1)
            result = st_math.linear_regression(X, y)
            coefs.append(result)
        return np.array(coefs)
    
    def adfuller_test(self) -> list:
        indexes = []
        for i in range(self.size, len(self.prices), self.size):
            result = adfuller(self.prices[i - self.window:i + self.size + self.window])
            if st_math.adfuller_check(result[4], result[1]):
                indexes.append(i)
        return indexes

    def panic_std_volatility(self) -> list:
        indexes = []
        for i in range(self.window + self.size, len(self.volatility), self.size):
            if max(self.volatility[i:i+self.size]) > np.mean(self.volatility) * self.panic_std_th:
                indexes.append(i - self.window)
        return indexes

    def panic_extremum(self) -> list:
        indexes = []
        for i in range(self.size + self.window, len(self.prices), self.size):
            if min(self.prices[i:i+self.size]) * self.panic_extr_th <= np.mean(self.prices[i-self.size:i]):
                indexes.append(i - self.window)
        return indexes
    
    def states(self) -> dict:
        states = {'bull': [], 'bear': [], 'stable': [], 'panic': [], 'undefined': []}
        pattern_aray = []
        bear_adx = self.bear_adx()
        bull_adx = self.bull_adx()
        bear_extr, bull_extr = self.extremum()
        static_adf_result = self.adfuller_test()
        linreg_coefs = self.linear_regression()
        panic_std = self.panic_std_volatility()
        panic_extr = self.panic_extremum()
        for i in range(self.size, len(self.prices), self.size):
            if i in panic_std or i in panic_extr:
                states['panic'].append(i)
                pattern_aray.append(-100)
                continue
            result = linreg_coefs[i // self.size - 1]
            if result > self.linreg_th:
                states['bull'].append(i)
                pattern_aray.append(1)
            elif 0 < result <= self.linreg_th and (i in bear_adx or i in bear_extr):
                states['bull'].append(i)
                pattern_aray.append(1)
            elif result < -self.linreg_th:
                states['bear'].append(i)
                pattern_aray.append(-1)
            elif 0 > result >= -self.linreg_th and (i in bull_adx or i in bull_extr):
                states['bear'].append(i)
                pattern_aray.append(-1)
            elif i in static_adf_result:
                states['stable'].append(i)
                pattern_aray.append(0)
            else:
                states['stable'].append(i)
                pattern_aray.append(0)
        return states, pattern_aray
