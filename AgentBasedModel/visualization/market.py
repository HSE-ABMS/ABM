from AgentBasedModel.simulator import SimulatorInfo
import AgentBasedModel.utils.math as math
import matplotlib.pyplot as plt


def plot_price(info: SimulatorInfo, spread=False, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Stock Price {info.exchange.id()}') if rolling == 1 else plt.title(
        f'Stock Price {info.exchange.id()} (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), color='black')
    if spread:
        v1 = [el['bid'] for el in info.spreads]
        v2 = [el['ask'] for el in info.spreads]
        plt.plot(range(rolling - 1, len(v1)), math.rolling(v1, rolling), label='bid', color='green')
        plt.plot(range(rolling - 1, len(v2)), math.rolling(v2, rolling), label='ask', color='red')
    plt.show()


def plot_price_fundamental(info: SimulatorInfo, spread=False, access: int = 1, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if rolling == 1:
        plt.title(f'Stock {info.exchange.id()} Fundamental and Market value')
    else:
        plt.title(f'Stock {info.exchange.id()} Fundamental and Market value (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Present value')
    if spread:
        v1 = [el['bid'] for el in info.spreads]
        v2 = [el['ask'] for el in info.spreads]
        plt.plot(range(rolling - 1, len(v1)), math.rolling(v1, rolling), label='bid', color='green')
        plt.plot(range(rolling - 1, len(v2)), math.rolling(v2, rolling), label='ask', color='red')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), label='market value', color='black')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.fundamental_value(access), rolling),
             label='fundamental value')
    plt.legend()
    plt.show()


def plot_arbitrage(info: SimulatorInfo, access: int = 1, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if rolling == 1:
        plt.title(f'Stock {info.exchange.id()} Fundamental and Market value difference %')
    else:
        plt.title(f'Stock {info.exchange.id()} Fundamental and Market value difference % (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Present value')
    market = info.prices
    fundamental = info.fundamental_value(access)
    arbitrage = [(fundamental[i] - market[i]) / fundamental[i] for i in range(len(market))]
    plt.plot(range(rolling, len(arbitrage)), math.rolling(arbitrage, rolling), color='black')
    plt.show()


def plot_dividend(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Stock {info.exchange.id()} Dividend') if rolling == 1 else plt.title(
        f'Stock {info.exchange.id()} Dividend (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Dividend')
    plt.plot(range(rolling, len(info.dividends)), math.rolling(info.dividends, rolling), color='black')
    plt.show()


def plot_orders(info: SimulatorInfo, stat: str = 'quantity', rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Book {info.exchange.id()} Orders') if rolling == 1 else plt.title(
        f'Book {info.exchange.id()} Orders (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel(stat)
    v1 = [v[stat]['bid'] for v in info.orders]
    v2 = [v[stat]['ask'] for v in info.orders]
    plt.plot(range(rolling, len(v1)), math.rolling(v1, rolling), label='bid', color='green')
    plt.plot(range(rolling, len(v2)), math.rolling(v2, rolling), label='ask', color='red')
    plt.legend()
    plt.show()


def plot_volatility_price(info: SimulatorInfo, window: int = 5, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Stock {info.exchange.id()} Price Volatility (window {window})')
    plt.xlabel('Iterations')
    plt.ylabel('Price Volatility')
    volatility = info.price_volatility(window)
    plt.plot(range(window, len(volatility) + window), volatility, color='black')
    plt.show()


def plot_volatility_return(info: SimulatorInfo, window: int = 5, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Stock {info.exchange.id()} Return Volatility (window {window})')
    plt.xlabel('Iterations')
    plt.ylabel('Return Volatility')
    volatility = info.return_volatility(window)
    plt.plot(range(window, len(volatility) + window), volatility, color='black')
    plt.show()


def plot_liquidity(info: SimulatorInfo, rolling: int = 1, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title(f'Liquidity {info.exchange.id()}') if rolling == 1 else plt.title(
        f'Liquidity {info.exchange.id()} (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Spread / avg. Price')
    plt.plot(info.liquidity(rolling), color='black')
    plt.show()


def plot_ema(info: SimulatorInfo, discount_factor: float = 0.94, rolling: int = 1, figsize=(6, 6)):
    prices = info.prices
    ema = []
    new_ema = []
    shock = False
    anchoring_ema = []
    for i in range(len(prices)):
        if i == 0:
            ema.append(prices[0])
        else:
            if abs(prices[i] - ema[-1]) > 10:
                shock = True
                new_ema[-1] = prices[i]
            ema.append(ema[-1] + discount_factor * (prices[i] - ema[-1]))
            if not shock:
                anchoring_ema.append(ema[-1])
                new_ema.append(ema[-1])
            else:
                new_ema.append(new_ema[-1] + discount_factor * (prices[i] - new_ema[-1]))
                anchoring_ema.append(anchoring_ema[-1] + 0.01 * (new_ema[-1] - anchoring_ema[-1]))

    plt.figure(figsize=figsize)
    plt.title(f'EMA {info.exchange.id()}') if rolling == 1 else plt.title(
        f'EMA {info.exchange.id()} (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('EMA')
    plt.plot(prices, color='black')
    plt.plot(ema, color='red')
    plt.plot(anchoring_ema, color='yellow')
    plt.plot(new_ema, color='orange')
    plt.show()
