from AgentBasedModel.simulator import SimulatorInfo
from AgentBasedModel.states import states_pattern
import AgentBasedModel.utils.states_math as st_math
from AgentBasedModel.states.states_pattern import StateIdentification
from AgentBasedModel.states.pattern import PatternDetection

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import time

class StatesVisualization:
    def __init__(self, info: SimulatorInfo = None, rolling: int = 1, figsize=(12, 6), size: int = 10, window: int = 5):
        if info is not None:
            self.prices = st_math.rolling(info.prices, rolling)
            self.volatility = info.price_volatility(window)
        self.rolling = rolling
        self.window = window
        self.size = size
        self.figsize = figsize

        self.state_idf = StateIdentification(info, rolling, size, window)
        self.pattern_det = PatternDetection()

    def set_state_params(self, adx_th: int = 20, max_th: float = 1.2, min_th: float = 1.2, linreg_th: float = 0.5, panic_std_th: float = 2.5, panic_extr_th: float = 1.5):
        self.adx_th = adx_th
        self.max_th = max_th
        self.min_th = min_th
        self.linreg_th = linreg_th
        self.panic_std_th = panic_std_th
        self.panic_extr_th = panic_extr_th

        self.state_idf.set_params(adx_th, max_th, min_th, linreg_th, panic_std_th, panic_extr_th)

    def set_pattern_params(self, local_eps: float = 1, local_min_samples: int = 5, local_pattern_size: int = 10, common_eps: float = 375, common_min_samples: int = 5):
        self.pattern_det.set_params(local_eps, local_min_samples, local_pattern_size, common_eps, common_min_samples)

    def start_plot(self, title: str, xlabel: str = 'Iterations', ylabel: str = 'Price'):
        plt.figure(figsize=self.figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    
    def plot_price(self):
        self.start_plot('Price')
        plt.plot(range(self.rolling - 1, len(self.prices)), self.prices, color='black')
        plt.show()

    def plot_volatility(self):
        self.start_plot('Volatility', ylabel='Volatility')
        plt.plot(range(self.window, len(self.volatility) + self.window), self.volatility, color='black')
        plt.show()

    def plot_adx(self):
        self.start_plot('Trends with ADX')

        bear_trend_adx = states_pattern.bear_trend_adx()
        bull_trend_adx = states_pattern.bull_trend_adx()

        plt.plot(bear_trend_adx, self.prices[bear_trend_adx], '.', color='green')
        plt.plot(bull_trend_adx, self.prices[bull_trend_adx], '.', color='red')
        plt.show()
    
    def plot_linreg(self):
        self.start_plot('Trends with Linear Regression')
        linreg_coefs = states_pattern.linear_regression_trends()
        bear_indexes = np.argwhere(linreg_coefs > self.linreg_th)
        bull_indexes = np.argwhere(linreg_coefs < -self.linreg_th)
        stable_indexes = np.argwhere(abs(linreg_coefs) <= self.linreg_th)
        plt.plot(bear_indexes, self.prices[bear_indexes], color='green')
        plt.plot(bull_indexes, self.prices[bull_indexes], color='red')
        plt.plot(stable_indexes, self.prices[stable_indexes], color='blue')
        plt.show()
    
    def plot_extremums(self):
        self.start_plot('Trends with extremums')

        bear_trend_extr, bull_trend_extr = states_pattern.trend_extremum()
        plt.plot(bear_trend_extr, self.prices[bear_trend_extr], '.', color='green')
        plt.plot(bull_trend_extr, self.prices[bull_trend_extr], '.', color='red')
        plt.show()
    
    def plot_adf_test(self):
        self.start_plot('Adfuller test')
        static_adf_result = states_pattern.adfuller_test(), self.size
        plt.plot(static_adf_result, self.prices[static_adf_result], '.', color='blue')
        plt.show()

    def plot_panic_std(self):
        self.start_plot('Panic with mean volatility')
        panic_std_indexes = states_pattern.panic_std(self.volatility)
        plt.plot(panic_std_indexes, self.prices[panic_std_indexes], '.', color='orange')
        plt.show()

    def plot_panic_extr(self):
        self.start_plot('Panic with local extremums')
        panic_extr_indexes = states_pattern.panic_extremum(self.prices), self.size
        plt.plot(panic_extr_indexes, self.prices[panic_extr_indexes], '.', color='orange')
        plt.show()

    def plot_moving_average(self):
        self.start_plot('Price and Moving Average')
        plt.plot(range(self.rolling - 1, len(self.prices)), self.prices, color='black')
        moving_average = st_math.moving_average(self.prices, self.window)
        plt.plot(range(self.window, len(moving_average) + self.window), moving_average, color='red')
        plt.legend(labels=['Price', 'SMA'])
        plt.show()
    
    def plot_adx(self):
        self.start_plot(title='Average Directional Index and Directional Movement Indicator', ylabel='Values')
        di_plus, di_minus, adx = st_math.average_directional_index(self.prices, self.window)
        plt.plot(range(len(adx)), adx, color='black')
        plt.plot(range(len(di_minus)), di_minus, color='red')
        plt.plot(range(len(di_plus)), di_plus, color='green')
        plt.legend(title='Indicators', labels=['ADX', '-DI', '+DI'])
        plt.show()

    def plot_states(self, prices = None):
        self.start_plot('States')
        states, _ = self.state_idf.states()
        colors = {'bull': 'green', 'bear': 'red', 'stable': 'blue', 'panic': 'orange', 'undefined': 'grey'}
        for state_name in states.keys():
            for index in states[state_name]:
                plt.plot(range(index-1, index + self.size), self.prices[index-1:index + self.size], color=colors[state_name])
        bear_patch = mpatches.Patch(color='red', label='Bear')
        bull_patch = mpatches.Patch(color='green', label='Bull')
        stable_patch = mpatches.Patch(color='blue', label='Stable')
        panic_patch = mpatches.Patch(color='orange', label='Panic')
        plt.legend(title='Types of states', handles=[bear_patch, bull_patch, stable_patch, panic_patch])
        plt.show()

    def plot_local_patterns(self, make_params: bool = False):
        start = time.time()
        windows_array = self.pattern_det.get_local_windows(self.state_idf)
        params = None
        if make_params:
            params = self.pattern_det.make_local_params()
            print(params)
        patterns, indexes = self.pattern_det.find_patterns(windows_array, params)
        if len(patterns) == 0:
            return
        self.start_plot('Local Patterns')
        plt.plot(range(self.rolling - 1, len(self.prices)), self.prices, color='black')
        n_clusters = len(patterns)
        colors = [plt.cm.inferno_r(i/float(n_clusters)) for i in range(n_clusters)]
        for cluster in range(len(indexes)):
            indexes[cluster] = (np.array(indexes[cluster])) * self.size
            for i_ in indexes[cluster]:
                i = i_[0]
                plt.plot(range(i, i + self.pattern_det.size + 1), self.prices[i:i + self.pattern_det.size + 1], color=colors[cluster])
        plt.show()

    def plot_common_patterns(self, simulations_info: list, make_params: bool = False):
        start = time.time()
        patterns = []
        simulations_states = []
        for info in simulations_info:
            state_idf = StateIdentification(info, self.rolling, self.size, self.window)
            state_idf.set_params(self.adx_th, self.max_th, self.min_th, self.linreg_th, self.panic_std_th, self.panic_extr_th)
            _, states_array = state_idf.states()
            simulations_states.append(states_array)

        params = None
        if make_params:
            params = self.pattern_det.make_common_params()
        patterns, indexes = self.pattern_det.find_patterns(np.array(simulations_states).reshape(-1, 1), params, type='common')
        if len(patterns) == 0:
            return
        self.start_plot('Common Patterns')
        colors = {1: 'green', -1: 'red', 0: 'blue', -100: 'orange'}
        bear_patch = mpatches.Patch(color='red', label='Bear')
        bull_patch = mpatches.Patch(color='green', label='Bull')
        stable_patch = mpatches.Patch(color='blue', label='Stable')
        panic_patch = mpatches.Patch(color='orange', label='Panic')
        for cluster in range(len(indexes)):
            info = simulations_info[indexes[cluster][0][0]]
            for i in range(self.size, len(info.prices), self.size):
                plt.plot(range(i - 1, i + self.size), info.prices[i - 1: i + self.size], color=colors[patterns[cluster][i // self.size - 1]])
            plt.legend(title='Types of states', handles=[bear_patch, bull_patch, stable_patch, panic_patch])
            print(time.time() - start)
            plt.show()
