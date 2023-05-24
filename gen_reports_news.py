import AgentBasedModel
import random

import json

from AgentBasedModel import plot_price_fundamental, plot_arbitrage, plot_price, plot_dividend, plot_orders
from AgentBasedModel.utils import logging
from AgentBasedModel import states

from os import listdir
from os.path import join

def generate_report(json_config_path):
    with open(json_config_path, "r", encoding="utf-8") as f:
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
    report = str(states.status(info))
    with open(json_config_path + "-out", "w", encoding="utf-8") as f:
        f.write(report)

for config in listdir("./experiments-news"):
    if config.endswith(".json"):
        generate_report(join("experiments-news", config))
