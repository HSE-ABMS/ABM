from AgentBasedModel import Simulator, Random, Fundamentalist, Chartist, ExchangeAgent, MarketMaker
from AgentBasedModel import *
import numpy as np
from AgentBasedModel.states.states_pattern import *
from AgentBasedModel.visualization.states import StatesVisualization
from AgentBasedModel.events.events import MarketPriceShock
import itertools

infos = []
def generateAgents(minVolume, maxVolume, exchangeCount):
    exchangesAgents = []
    for _ in range(exchangeCount):
        exchangesAgents.append(ExchangeAgent(volume=500))
    return exchangesAgents


# Common patterns
# for it in itertools.product(2**np.arange(4), repeat=4):
it = [1, 1, 1, 1]
exchanges = [ExchangeAgent()]
traders = []
traders.extend([Random(exchanges, cash=1000, assets=[0]) for _ in range(it[0])])
# traders += [Fundamentalist(xg, cash=1000, assets=[0]) for _ in range(it[1])]
# traders += [Chartist(xg, cash=1000, assets=[0]) for _ in range(it[2])]
# traders += [MarketMaker(xg, cash=1000, assets=[0]) for _ in range(it[3])]

simulator = Simulator(**{
    'exchanges': exchanges,
    'traders': traders,
    'events': [MarketPriceShock(200, -10)],
})

try:
    simulator.simulate(100)
except Exception as e:
    print(e)
    # continue
info = simulator.info[0]
infos.append(info)

# vis = StatesVisualization()
# vis.set_state_params()
# vis.set_pattern_params()
# vis.plot_common_patterns(infos, True)
