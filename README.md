# ABM
Agent-Based Model that simulates trading on a financial market.
The underlying paper for this model is below.

Thesis link: https://www.hse.ru/en/edu/vkr/639867359

## Getting Started
Here is a quick snippet how to launch the simulation and obtain price graph.

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
3. Allow traders to change their strategy if needed.
4. Call traders for actions.
5. Pay risk-free cash income, and dividends.

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