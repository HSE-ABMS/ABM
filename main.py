from AgentBasedModel import *
from AgentBasedModel.visualization import states, other

exchange = ExchangeAgent(volume=1000)
simulator = Simulator(**{
    'exchange': exchange,
    'traders': [Universalist(exchange, 10**3) for _ in range(20)],
    'events': [MarketPriceShock(200, -10)]
})
info = simulator.info
simulator.simulate(500)

print(info.exchange.order_history)

# plot_price_fundamental(info)
# plot_price(info)
# plot_arbitrage(info) не работает
# plot_dividend(info) не работает
# plot_orders(info) не работает
# plot_volatility_price(info)
# plot_volatility_return(info)
# plot_liquidity(info)
# other.plot_book(info)
# states.plot_exp_moving_average(info)