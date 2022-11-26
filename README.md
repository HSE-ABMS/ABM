# ABM
Agent-Based Model that simulates trading on a financial market.

For a more specific info about the project welcome to the thesis text:
https://www.hse.ru/en/edu/vkr/639867359

## Getting Started
Here is a quick snippet how to launch the simulation and obtain price graph:

    from AgentBasedModel import *

    exchange = ExchangeAgent(volume=1000)
    simulator = Simulator(**{
        'exchange': exchange,
        'traders': [Universalist(exchange, 10**3) for _ in range(20)]
    })
    info = simulator.info
    simulator.simulate(500)
    
    plot_price(info)

This code simulates single stock market with traders that can use both **fundamental**
and **speculative** strategies for 500 iterations and plots its price graph.

### Simulation dynamics

The trading simulation is realised through a number of iterative trading
sessions. On each session agents are called to undertake their actions
according to their strategy. The agents can access any ‘public’
information from the exchange agent, as each of them is granted with a
link to it upon initialisation. Such information could be stock price,
best bid, and best ask prices, values for future dividends depending on
trader’s access.

**Trading session sequencing**
1. Launch events if any hit their activation time at the iteration.
2. Capture market, and traders’ stats.
3. Opinion revaluation / change in strategy
4. Call traders for actions.
5. Pay risk-free cash income, and dividends.

## Agents

**Exchange Agent** - represents both stock exchange and stock itself.
It handles assets and cash exchange between traders, generate next dividends,
maintains  order book, returns *visible* statistics about stock and market.
When initialised, fill the order book with random orders. Order book
management is enhanced with linked list data structure.

**Random** - trader that creates random orders. Represents individual
traders without specific trading strategies.

**Fundamentalist** - evaluate stock value using Constant Dividend Model.
Then places orders accordingly

**Chartist** - search for trends in price movements. Each trader has
the sentiment - opinion  about future price movement (either increasing,
or decreasing). Based on sentiment trader either  buys stock or sells.
Sentiment revaluation happens at the end of each iteration based on opinion
propagation among other chartists, current price changes.

**Universalist** - combine both Fundamentalist and Chartist allowing
to choose between these two strategies based on relative trader's
performance and market condition.

**Market Maker** - creates limit orders on both sides of the spread
trying to gain on the spread between bid and ask prices, and maintains
its assets to cash ratio in balance.

## Project Structure

- !**agents** - all Traders + ExchangeAgent definitions
- **events** - all Events that affect market
- !**simulator** - Simulator and SimulatorInfo definition (these
  are used to handle the simulation sequencing and statistics
  capture)
- **states** - these are market states and transitions capturing (don't
  mess with it)
- **utils** - additional useful functions defined (such as math functions,
  order handling data structures for ***ExchangeAgent***)
- !**visualisation** - matplotlib graph functions, separated by meaning:
  *market* - market statistics plot (such as stock price, volatility, etc.),
  *trader* - trader's specific statistics (such as sum of cash, equities
  of all traders, their returns)

*! - important stuff*