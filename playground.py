from AgentBasedModel import Simulator, ExchangeAgent, Random, Fundamentalist, Chartist

xg = ExchangeAgent()
traders = []
traders += [Random(xg, cash=1000) for i in range(10)]
traders += [Fundamentalist(xg, cash=1000) for i in range(10)]
traders += [Chartist(xg, cash=1000) for i in range(10)]
sim = Simulator(exchange=ExchangeAgent(), traders=traders)
sim.simulate(100)
