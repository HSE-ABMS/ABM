from AgentBasedModel.simulator import SimulatorInfo
from AgentBasedModel.states import states_pattern

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import AgentBasedModel.utils.math as math

def plot_moving_average(info: SimulatorInfo, rolling: int = 1, figsize=(12, 6), size: int = 25):
    plt.figure(figsize=figsize)
    plt.title('Moving Average') if rolling == 1 else plt.title(f'Moving Average (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    moving_average_25 = states_pattern.moving_average(info.prices, 25)
    moving_average_50 = states_pattern.moving_average(info.prices, 50)
    moving_average_100 = states_pattern.moving_average(info.prices, 100)
    moving_average_200 = states_pattern.moving_average(info.prices, 200)
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), color='black')
    plt.plot(range(rolling - 1, len(moving_average_25)), moving_average_25, color='red')
    plt.plot(range(rolling - 1, len(moving_average_50)), moving_average_50, color='green')
    plt.plot(range(rolling - 1, len(moving_average_100)), moving_average_100, color='orange')
    plt.plot(range(rolling - 1, len(moving_average_200)), moving_average_200, color='blue')
    plt.show()

def plot_exp_moving_average(info: SimulatorInfo, rolling: int = 1, figsize=(12, 6), size: int = 25):
    plt.figure(figsize=figsize)
    plt.title('Moving Average') if rolling == 1 else plt.title(f'Moving Average (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    exp_moving_average_25 = states_pattern.moving_average(info.prices, 25)
    # plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), color='black')
    orders = info.exchange.order_book['bid'].to_list()
    print(orders[0])
    res = []
    for a in orders:
        if a['order_type'] == 'bid':
            res.append(a['price'])
    # plt.plot(range(rolling - 1, len(exp_moving_average_25)), exp_moving_average_25, color='red')
    plt.plot(range(rolling - 1, len(res)), res, color='red')
    # plt.plot(range(rolling - 1, len(orders['ask'])), orders['ask'], color='green')
    plt.show()

# def plot_moving_order_book(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
