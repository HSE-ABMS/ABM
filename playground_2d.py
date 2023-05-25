import json

from AgentBasedModel.visualization.market import plot_price_fundamental, plot_arbitrage, plot_price, \
    plot_dividend, plot_orders
from AgentBasedModel.visualization.trader import plot_equity
from AgentBasedModel.utils import logging, loader


with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
logging.Logger.info(f"Config loaded: {json.dumps(config)}")

simulator = loader.load_simulator(config)
simulator.simulate(config["iterations"])

info = simulator.info
for _ in range(len(info)):
    plot_price_fundamental(info[_])
    plot_arbitrage(info[_])
    plot_price(info[_])
    plot_dividend(info[_])
    plot_orders(info[_])

    plot_equity(info[_])
