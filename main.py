import imp
import AgentBasedModel
imp.reload(AgentBasedModel)

from AgentBasedModel.visualization import states, states_other, market
from AgentBasedModel.states import states_pattern

exchange = AgentBasedModel.ExchangeAgent(volume=1000)
simulator = AgentBasedModel.Simulator(**{
    'exchange': exchange,
    'traders': [AgentBasedModel.Universalist(exchange, 10**3) for _ in range(20)],
    'events': [AgentBasedModel.MarketPriceShock(200, -10), AgentBasedModel.MarketPriceShock(500, +10)]
})
info = simulator.info
simulator.simulate(1000)

# states.plot_bear_trend_all_indicators(info)
# states.plot_trends_local_extremums(info)
# states_other.plot_moving_average(info)
# states_other.plot_exp_moving_average(info)
# states_other.plot_adx(info)
# states_other.plot_standart_moving_average(info)
# states.plot_trends_linreg(info)
states.plot_trends(info)
# states.plot_panic(info)
# states_other.plot_adfuller(info)
# states.plot_trends_local_extremums(info)
# market.plot_volatility_price(info)
# market.plot_orders(info)
# market.plot_liquidity(info)
# market.plot_price(info)
