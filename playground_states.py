from AgentBasedModel import Simulator, Random, Fundamentalist, Chartist, ExchangeAgent, MarketMaker
from AgentBasedModel import *
import numpy as np
from AgentBasedModel.states.states_pattern import *
from AgentBasedModel.visualization.states import StatesVisualization
from AgentBasedModel.events.events import MarketPriceShock
import itertools

# SIMULATOR INFO

xg = ExchangeAgent()
traders = []
traders += [Random([xg], cash=1000) for i in range(10)]
traders += [Fundamentalist([xg], cash=1000) for i in range(10)]
traders += [Chartist([xg], cash=1000) for i in range(10)]
traders += [MarketMaker([xg], cash=1000) for i in range(10)]

sim = Simulator(exchanges=[xg], traders=traders, events=[MarketPriceShock(200, -10), TransactionCost(500, 5)])

try:
    sim.simulate(1000)
except Exception as e:
    print(e)
    exit()

vis = StatesVisualization(sim.info[0])
vis.set_state_params()
vis.set_pattern_params()

# SIMULATOR INFO
# vis.plot_price()
# vis.plot_volatility()
# vis.plot_moving_average()

# INDICATORS AND OTHER METHODS
# vis.plot_adx()
# vis.plot_linreg()
# vis.plot_extremums()
# vis.plot_adf_test()
# vis.plot_panic_std()

# ALL STATES
# vis.plot_states()

# LOCAL PATTERNS
# vis.plot_local_patterns(True)

# COMMON PATTERNS
infos = []
for it in itertools.product(2**np.arange(2), repeat=4):
    xg = ExchangeAgent()
    traders = []
    traders += [Random([xg], cash=1000) for i in range(it[0])]
    traders += [Fundamentalist([xg], cash=1000) for i in range(it[1])]
    traders += [Chartist([xg], cash=1000) for i in range(it[2])]
    traders += [MarketMaker([xg], cash=1000) for i in range(it[3])]

    sim = Simulator(exchanges=[xg], traders=traders, events=[MarketPriceShock(200, -10)])
    sim.simulate(500)

    try:
        sim.simulate(100)
    except Exception as e:
        print(e)
        continue
    info = sim.info[0]
    infos.append(info)

vis = StatesVisualization()
vis.set_state_params()
vis.set_pattern_params()
vis.plot_common_patterns(infos, True)
