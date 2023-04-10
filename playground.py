from AgentBasedModel import Simulator, ExchangeAgent, Random, Fundamentalist, Chartist

xg = ExchangeAgent()
traders = []
traders += [Random(xg, cash=1000) for i in range(10)]
traders += [Fundamentalist(xg, cash=1000) for i in range(10)]
traders += [Chartist(xg, cash=1000) for i in range(10)]

sim = Simulator(exchange=xg, traders=traders)

sim.simulate(100000)

sim.info.prices
import numpy as np
np.std(np.array(sim.info.prices))
sim.info.orders

import matplotlib.pyplot as plt
plt.plot(np.array(sim.info.prices) + 222)
plt.savefig('/tmp/out.png')
