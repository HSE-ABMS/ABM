from AgentBasedModel import Simulator, ExchangeAgent, Random
from AgentBasedModel import NumericalFundamentalist
from AgentBasedModel import NumericalNews, CategoricalNews, News
import numpy as np

xg = ExchangeAgent()
traders = []
traders += [Random(xg, cash=1000, assets=10) for i in range(10)]
traders += [NumericalFundamentalist(xg, float(75 + 40 * i), 10 * i, cash=10000, assets=100) for i in range(2)]

sim = Simulator(exchange=xg, traders=traders)

news = [
    # (10, NumericalNews())
]
sim.simulate(100, news)

sim.info.prices
np.std(np.array(sim.info.prices))
sim.info.orders

import matplotlib.pyplot as plt
plt.plot(np.array(sim.info.prices) + 222)
plt.savefig('/tmp/out.png')


a = 5
type(a) is int

