from AgentBasedModel import Simulator, ExchangeAgent, Random
from AgentBasedModel import NumericalFundamentalist, AdaptiveNumericalFundamentalist
from AgentBasedModel import NumericalNews, CategoricalNews, News, InfoFlow
from AgentBasedModel import NoiseGenerator
import numpy as np
from random import random, randint

xg = ExchangeAgent()
traders = []
# traders += [Random(xg, cash=1000, assets=10) for i in range(10)]
# traders += [NumericalFundamentalist(xg, float(75 + 40 * i), 10 * i, cash=10000, assets=100) for i in range(2)]
traders += [NumericalFundamentalist(expectation=random() * 40 + 80, delay=randint(0, 12), market=xg, cash=1000, assets=100) for i in range(100)]
traders += [AdaptiveNumericalFundamentalist(phi=0.01, expectation=random() * 60 + 70, delay=randint(0, 12), market=xg, cash=1000, assets=100) for i in range(10)]

sim = Simulator(exchange=xg, traders=traders)

gen = NoiseGenerator()
news = InfoFlow()
step = 0
for i in range(100):
    news.put(randint(60, 120), NumericalNews(gen.next() + 100))
sim.simulate(10000, news)

sim.info.prices
np.std(np.array(sim.info.prices))
sim.info.orders

import matplotlib.pyplot as plt
plt.plot(np.array(sim.info.prices))
plt.savefig('/tmp/out.png')

