from AgentBasedModel import Simulator, ExchangeAgent, Random, Fundamentalist, Chartist

xg = ExchangeAgent()
traders = []
traders += [Random(xg, cash=1000) for i in range(10)]
traders += [Fundamentalist(xg, cash=1000) for i in range(10)]
traders += [Chartist(xg, cash=1000) for i in range(10)]

sim = Simulator(exchange=xg, traders=traders)

sim.simulate(1000)

import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.array(sim.info.prices))
plt.savefig('/tmp/out.png', dpi=250)

sim.info.prices
np.std(np.array(sim.info.prices))
sim.info.orders

