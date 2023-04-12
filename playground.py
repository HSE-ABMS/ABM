from AgentBasedModel import Simulator, ExchangeAgent, Random, FakeExchangeAgent
from AgentBasedModel import NumericalFundamentalist, AdaptiveNumericalFundamentalist
from AgentBasedModel import NumericalNews, CategoricalNews, News, InfoFlow
from AgentBasedModel import NoiseGenerator
import numpy as np
from random import random, randint, seed

seed(42)
xg = ExchangeAgent()
traders = []
traders += [Random(xg, cash=1000, assets=10) for i in range(100)]
fxg = FakeExchangeAgent()
traders += [NumericalFundamentalist(expectation=random() * 40 + 80, delay=randint(0, 2), market=fxg, cash=1000, assets=10) for i in range(100)]
traders += [AdaptiveNumericalFundamentalist(phi=0.01, expectation=random() * 60 + 70, delay=randint(0, 12), market=fxg, cash=1000, assets=100) for i in range(10)]

sim = Simulator(exchange=xg, traders=traders)

gen = NoiseGenerator()
news = InfoFlow()
step = 0
news_step = []
for i in range(30):
    delay = randint(60, 120)
    step += delay
    news_step.append(step)
    news.put(delay, NumericalNews(gen.next() + 100))
news.put(0, NumericalNews(1000))
sim.simulate(3000, news)

import matplotlib.pyplot as plt
arr = np.array(sim.info.prices)
plt.figure(figsize=(20, 10), dpi=200)
plt.plot(arr, label='price')
plt.plot(news_step, arr[news_step], linestyle='None', marker='o', label='news')
plt.legend()
plt.savefig('/tmp/out.png')

