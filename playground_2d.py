from AgentBasedModel import *
import random

from AgentBasedModel.events.events import LiquidityShock


def generateAgents(minVolume, maxVolume, exchangeCount):
    exchangesAgents = []
    for _ in range(exchangeCount):
        exchangesAgents.append(Broker(volume=random.randint(minVolume, maxVolume)))
    return exchangesAgents


exchanges = generateAgents(300, 2000, 2)

traders = []

traders.extend(
    [
        Universalist([exchanges[random.randint(0, len(exchanges) - 1)]], 10 ** 3, [0]) for _ in range(500)
    ]
)
traders.extend(
    [
        Fundamentalist([exchanges[random.randint(0, len(exchanges) - 1)]], 10 ** 3, [0]) for _ in range(100)
    ]
)
traders.extend(
    [
        Chartist(exchanges, 10 ** 3, [0, 0]) for _ in range(10)
    ]
)

simulator = Simulator(**{
    'exchanges': exchanges,
    'traders': traders,
    'events': [MarketPriceShock(200, -10)],
})
simulator.simulate(500)

info = simulator.info
for _ in range(len(info)):
    plot_price_fundamental(info[_])
