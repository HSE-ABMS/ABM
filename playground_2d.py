import AgentBasedModel
import random

import json

from AgentBasedModel import plot_price_fundamental, plot_arbitrage, plot_price, plot_dividend, plot_orders
from AgentBasedModel.utils import logging

with open("config.json", "r", encoding="utf-8") as f:
    config = json.loads(f.read())
logging.Logger.info(f"Config: {json.dumps(config)}")
exchanges = []
traders = []
events = []

for exchange in config["exchanges"]:
    exchanges.append(AgentBasedModel.ExchangeAgent(**exchange))
for trader in config["traders"]:
    params = dict(**trader)
    params.pop("type")
    params.pop("count")
    params["markets"] = [exchanges[_] for _ in trader["markets"]]
    traders.extend(
        [
            getattr(AgentBasedModel.agents, trader["type"])(**params) for _ in range(trader["count"])
        ]
    )
for event in config["events"]:
    params = dict(**event)
    params.pop("type")
    events.append(
        getattr(AgentBasedModel.events, event["type"])(**params)
    )
simulator = AgentBasedModel.Simulator(**{
    'exchanges': exchanges,
    'traders': traders,
    'events': events,
})
simulator.simulate(config["iterations"])

info = simulator.info
for _ in range(len(info)):
    plot_price_fundamental(info[_])
    plot_arbitrage(info[_])
    plot_price(info[_])
    plot_dividend(info[_])
    plot_orders(info[_])
