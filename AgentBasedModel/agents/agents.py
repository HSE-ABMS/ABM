import queue
from typing import List, Dict

from AgentBasedModel.utils import Order, OrderList, logging
from AgentBasedModel.utils.math import exp, mean
from AgentBasedModel.news import InfoFlow
import random
from abc import abstractmethod
from queue import Queue
from AgentBasedModel.news.news import News, CategoricalNews, NumericalNews
from math import log


class Broker:
    """
    Broker implements automatic orders handling within the order book
    """

    def _not_impl(_):
        raise Exception("Not implemented")

    def __init__(self, price: float or int = 100, std: float or int = 25, volume: int = 1000, rf: float = 5e-4,
                 transaction_cost: float = 0):
        """
        Creates ExchangeAgent with initialised order book and future dividends

        :param price: stock initial price
        :param std: standard deviation of order prices
        :param volume: number of orders initialised
        :param rf: risk-free rate
        :param transaction_cost: transaction cost on operations for traders
        """
        self._not_impl()

    def generate_dividend(self):
        """
        Add new dividend to queue and pop last
        """
        self._not_impl()

    def spread(self) -> dict or None:
        """
        Returns best bid and ask prices as dictionary

        :return: {'bid': float, 'ask': float}
        """
        self._not_impl()

    def spread_volume(self) -> dict or None:
        """
        **(UNUSED)** Returns best bid and ask volumes as dictionary

        :return: {'bid': float, 'ask': float}
        """
        self._not_impl()

    def price(self) -> float:
        """
        Returns current stock price as mean between best bid and ask prices
        If price cannot be determined, None is returned
        """
        self._not_impl()

    def dividend(self, access: int = None) -> list or float:
        """
        Returns either current dividend or *access* future dividends (if called by trader)

        :param access: the number of future dividends accessed by a trader
        """
        self._not_impl()

    def limit_order(self, order: Order):
        self._not_impl()

    def market_order(self, order: Order) -> Order:
        self._not_impl()

    def cancel_order(self, order: Order):
        self._not_impl()

    def id(self) -> int:
        self._not_impl()

    def order_book(self) -> Dict[str, OrderList]:
        self._not_impl()

    def risk_free(self) -> float:
        self._not_impl()

    def iteration(self) -> int:
        self._not_impl()

    def increment_iteration(self):
        self._not_impl()

    def transaction_cost(self) -> float:
        self._not_impl()

    def set_transaction_cost(self, cost):
        self._not_impl()


class ExchangeAgent(Broker):
    global_id = 0

    def __init__(self, price: float or int = 100, std: float or int = 25, volume: int = 1000, rf: float = 5e-4,
                 transaction_cost: float = 0):
        self._id = ExchangeAgent.global_id
        self._iteration = 0
        self.name = f'ExchangeAgent{self.id}'
        ExchangeAgent.global_id += 1
        self.volume = volume
        self._order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
        self.dividend_book = list()  # act like queue
        self._risk_free = rf
        self._transaction_cost = transaction_cost
        self._fill_book(price, std, volume, rf * price)  # initialise both order book and dividend book
        logging.Logger.info(f"{self.name}")

    def generate_dividend(self):
        # Generate future dividend
        d = self.dividend_book[-1] * self._next_dividend()
        self.dividend_book.append(max(d, 0))  # dividend > 0
        self.dividend_book.pop(0)

    def _fill_book(self, price, std, volume, div: float = 0.05):
        """
        Fill order book with random orders and fill dividend book with future dividends.

        **Order book:** generate prices from normal distribution with *std*, and *price* as center; generate
        quantities for these orders from uniform distribution with 1, 5 bounds. Set bid orders when
        order price is less than *price*, set ask orders when order price is greater than *price*.

        **Dividend book:** add 100 dividends using *_next_dividend* method.
        """
        # Order book
        prices1 = [round(random.normalvariate(price - std, std), 1) for _ in range(volume // 2)]
        prices2 = [round(random.normalvariate(price + std, std), 1) for _ in range(volume // 2)]
        quantities = [random.randint(1, 5) for _ in range(volume)]

        for (p, q) in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = Order(round(p, 1), q, 'ask', 0, None)
                self._order_book['ask'].append(order)
            else:
                order = Order(p, q, 'bid', 0, None)
                self._order_book['bid'].push(order)

        # Dividend book
        for i in range(100):
            self.dividend_book.append(max(div, 0))  # dividend > 0
            div *= self._next_dividend()

    def _clear_book(self):
        """
        **(UNUSED)** Clear order book from orders with 0 quantity.
        """
        self._order_book['bid'] = OrderList.from_list([order for order in self._order_book['bid'] if order.qty > 0])
        self._order_book['ask'] = OrderList.from_list([order for order in self._order_book['ask'] if order.qty > 0])

    def spread(self) -> dict or None:
        if self._order_book['bid'] and self._order_book['ask']:
            return {'bid': self._order_book['bid'].first.price, 'ask': self._order_book['ask'].first.price}
        # raise Exception(f'There no either bid or ask orders')
        return None

    def spread_volume(self) -> dict or None:
        if self._order_book['bid'] and self._order_book['ask']:
            return {'bid': self._order_book['bid'].first.qty, 'ask': self._order_book['ask'].first.qty}
        # raise Exception(f'There no either bid or ask orders')
        return None

    def price(self) -> float or None:
        spread = self.spread()
        if spread:
            return round((spread['bid'] + spread['ask']) / 2, 1)
        # raise Exception(f'Price cannot be determined, since no orders either bid or ask')
        return None

    def dividend(self, access: int = None) -> list or float:
        if access is None:
            return self.dividend_book[0]
        return self.dividend_book[:access]

    @classmethod
    def _next_dividend(cls, std=5e-3):
        return exp(random.normalvariate(0, std))

    def limit_order(self, order: Order):
        spread = self.spread()
        if spread is None:
            return
        bid, ask = spread.values()
        t_cost = self.transaction_cost()
        if not bid or not ask:
            return

        if order.order_type == 'bid':
            if order.price >= ask:
                order = self._order_book['ask'].fulfill(order, t_cost)
            if order.qty > 0:
                self._order_book['bid'].insert(order)
            return

        elif order.order_type == 'ask':
            if order.price <= bid:
                order = self._order_book['bid'].fulfill(order, t_cost)
            if order.qty > 0:
                self._order_book['ask'].insert(order)

    def market_order(self, order: Order) -> Order:
        t_cost = self.transaction_cost()
        if order.order_type == 'bid':
            order = self._order_book['ask'].fulfill(order, t_cost)
        elif order.order_type == 'ask':
            order = self._order_book['bid'].fulfill(order, t_cost)
        return order

    def cancel_order(self, order: Order):
        if order.order_type == 'bid':
            self._order_book['bid'].remove(order)
        elif order.order_type == 'ask':
            self._order_book['ask'].remove(order)

    def order_book(self):
        return self._order_book

    def risk_free(self):
        return self._risk_free

    def iteration(self):
        return self._iteration

    def increment_iteration(self):
        self._iteration += 1

    def transaction_cost(self):
        return self._transaction_cost

    def set_transaction_cost(self, cost):
        self._transaction_cost = cost

    def id(self):
        return self._id


class Trader:
    """
    Trader basic interface
    """
    id = 0

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int]):
        """
        Trader that is activated on call to perform action

        :param markets: link to exchange agent
        :param cash: trader's cash available
        :param assets: trader's number of shares hold
        """
        self.type = 'Unknown'
        self.name = f'Trader{self.id}'
        self.id = Trader.id
        Trader.id += 1

        self.markets = markets
        self.orders = list()  # list of orders sitting in the order book

        self.cash = cash
        self.assets = assets
        logging.Logger.info(f"{self.name} {len(assets)}")

    def __str__(self) -> str:
        return f'{self.name} ({self.type})'

    def equity(self) -> float:
        """
        Returns trader's value of cash and stock holdings
        """
        price = 0
        for _ in range(len(self.markets)):
            price += self.assets[_] * (self.markets[_].price() if self.markets[_].price() is not None else 0)
        return self.cash + price

    def _buy_limit(self, quantity, price, market_id):
        order = Order(round(price, 1), round(quantity), 'bid', market_id, self.markets[market_id].iteration(), self)
        self.orders.append(order)
        self.markets[market_id].limit_order(order)

    def _sell_limit(self, quantity, price, market_id):
        order = Order(round(price, 1), round(quantity), 'ask', market_id, self.markets[market_id].iteration(), self)
        self.orders.append(order)
        self.markets[market_id].limit_order(order)

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        for _ in range(len(self.markets)):
            if self.markets[_].order_book()['ask']:
                break
        else:
            return quantity
        mn_index = -1
        for _ in range(len(self.markets)):
            if self.markets[_].order_book()['ask'].last is None:
                continue
            if mn_index == -1 or self.markets[_].order_book()['ask'].last.price < self.markets[mn_index].order_book()['ask'].last.price:
                mn_index = _
        if mn_index == -1:
            return quantity
        logging.Logger.info(f"{self.name} ({self.type}) BUY {mn_index}/{len(self.markets)}")
        order = Order(self.markets[mn_index].order_book()['ask'].last.price, round(quantity), 'bid', mn_index,
                      self.markets[mn_index].iteration(), self)
        return self.markets[mn_index].market_order(order).qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        for _ in range(len(self.markets)):
            if self.markets[_].order_book()['bid']:
                break
        else:
            return quantity
        mn_index = -1
        for _ in range(len(self.markets)):
            if self.markets[_].order_book()['bid'].last is None:
                continue
            if mn_index == -1 or self.markets[_].order_book()['bid'].last.price > self.markets[mn_index].order_book()['bid'].last.price:
                mn_index = _
        if mn_index == -1:
            return quantity
        logging.Logger.info(f"{self.name} ({self.type}) SELL {mn_index}/{len(self.markets)}")
        order = Order(self.markets[mn_index].order_book()['bid'].last.price, round(quantity), 'ask', mn_index,
                      self.markets[mn_index].iteration(), self)
        return self.markets[mn_index].market_order(order).qty

    def _cancel_order(self, order: Order):
        self.markets[order.market_id].cancel_order(order)
        self.orders.remove(order)

    @abstractmethod
    def refresh(self, info):
        pass

    @abstractmethod
    def call(self):
        pass


class Random(Trader):
    """
    Random creates noisy orders to recreate trading in real environment.
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Random'

    @staticmethod
    def draw_delta(std: float or int = 2.5):
        lamb = 1 / std
        return random.expovariate(lamb)

    @staticmethod
    def draw_price(order_type, spread: dict, std: float or int = 2.5) -> float:
        """
        Draw price for limit order. The price is calculated as:

        - 0.35 probability: within the spread - uniform distribution
        - 0.65 probability: out of the spread - delta from best price is exponential distribution r.v.
        :return: order price
        """
        random_state = random.random()  # Determines IN spread OR OUT of spread

        # Within the spread
        if random_state < .35:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = Random.draw_delta(std)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    @staticmethod
    def draw_quantity(a=1, b=5) -> float:
        """
        Draw order quantity for limit or market order. The quantity is drawn from U[a,b]

        :param a: minimal quantity
        :param b: maximal quantity
        :return: order quantity
        """
        return random.randint(a, b)

    def call(self):
        spread = list(filter(lambda x: x is not None, [self.markets[_].spread() for _ in range(len(self.markets))]))
        if len(spread) == 0:
            return
        spread = min(spread)
        random_state = random.random()

        if random_state > .5:
            order_type = 'bid'
        else:
            order_type = 'ask'

        random_state = random.random()
        # Market order
        if random_state > .85:
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order
        elif random_state > .5:
            price = self.draw_price(order_type, spread)
            quantity = self.draw_quantity()
            if order_type == 'bid':
                self._buy_limit(quantity, price, 0)
            elif order_type == 'ask':
                self._sell_limit(quantity, price, 0)

        # Cancellation order
        elif random_state < .35:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class Fundamentalist(Trader):
    """
    Fundamentalist evaluate stock value using Constant Dividend Model. Then places orders accordingly
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None, access: int = 1):
        """
        :param markets: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Fundamentalist'
        self.access = access

    @staticmethod
    def evaluate(dividends: list, risk_free: float):
        """
        Evaluates the stock using Constant Dividend Model.

        We first sum all known (based on *access*) discounted dividends. Then calculate perpetual value
        of the stock based on last known dividend.
        """
        divs = dividends  # known future dividends
        r = risk_free  # risk-free rate

        perp = divs[-1] / r / (1 + r) ** (len(divs) - 1)  # perpetual value
        known = sum([divs[i] / (1 + r) ** (i + 1) for i in range(len(divs) - 1)]) if len(divs) > 1 else 0  # known value
        return known + perp

    @staticmethod
    def draw_quantity(pf, p, gamma: float = 5e-3):
        """
        Draw order quantity for limit or market order. The quantity depends on difference between fundamental
        value and market price.

        :param pf: fundamental value
        :param p: market price
        :param gamma: dependency coefficient
        :return: order quantity
        """
        q = round(abs(pf - p) / p / gamma)
        return min(q, 5)

    def call(self):
        pf = round(self.evaluate(self.markets[0].dividend(self.access), self.markets[0].risk_free()),
                   1)  # fundamental price
        p = self.markets[0].price()
        spread = self.markets[0].spread()
        t_cost = self.markets[0].transaction_cost()

        if spread is None:
            return

        random_state = random.random()
        qty = Fundamentalist.draw_quantity(pf, p)  # quantity to buy
        if not qty:
            return

        # Limit or Market order
        if random_state > .45:
            random_state = random.random()

            ask_t = round(spread['ask'] * (1 + t_cost), 1)
            bid_t = round(spread['bid'] * (1 - t_cost), 1)

            if pf >= ask_t:
                if random_state > .5:
                    self._buy_market(qty)
                else:
                    self._sell_limit(qty, (pf + Random.draw_delta()) * (1 + t_cost), 0)

            elif pf <= bid_t:
                if random_state > .5:
                    self._sell_market(qty)
                else:
                    self._buy_limit(qty, (pf - Random.draw_delta()) * (1 - t_cost), 0)

            elif ask_t > pf > bid_t:
                if random_state > .5:
                    self._buy_limit(qty, (pf - Random.draw_delta()) * (1 - t_cost), 0)
                else:
                    self._sell_limit(qty, (pf + Random.draw_delta()) * (1 + t_cost), 0)

        # Cancel order
        else:
            if self.orders:
                self._cancel_order(self.orders[0])


class Chartist(Trader):
    """
    Chartists are searching for trends in the price movements. Each trader has sentiment - opinion
    about future price movement (either increasing, or decreasing). Based on sentiment trader either
    buys stock or sells. Sentiment revaluation happens at the end of each iteration based on opinion
    propagation among other chartists, current price changes
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None):
        """
        :param markets: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Chartist'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'

    def call(self):
        random_state = random.random()

        if self.sentiment == 'Optimistic':
            mn_index = 0
            for _ in range(len(self.markets)):
                if self.markets[_].price() < self.markets[mn_index].price():
                    mn_index = _
            t_cost = self.markets[mn_index].transaction_cost()
            spread = self.markets[mn_index].spread()
            # Market order
            if random_state > .85:
                self._buy_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                self._buy_limit(Random.draw_quantity(), Random.draw_price('bid', spread) * (1 - t_cost), mn_index)
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])
        elif self.sentiment == 'Pessimistic':
            mx_index = 0
            for _ in range(len(self.markets)):
                if self.markets[_].price() < self.markets[mx_index].price():
                    mx_index = _
            t_cost = self.markets[mx_index].transaction_cost()
            spread = self.markets[mx_index].spread()
            # Market order
            if random_state > .85:
                self._sell_market(Random.draw_quantity())
            # Limit order
            elif random_state > .5:
                self._sell_limit(Random.draw_quantity(), Random.draw_price('ask', spread) * (1 + t_cost), mx_index)
            # Cancel order
            elif random_state < .35:
                if self.orders:
                    self._cancel_order(self.orders[-1])

    def change_sentiment(self, info, a1=1, a2=1, v1=.1):
        """
        Revaluate chartist's opinion about future price movement

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param v1: frequency of revaluation of opinion for sentiment
        """
        n_traders = len(info.traders)  # number of all traders
        n_chartists = sum([tr_type == 'Chartist' for tr_type in info.types[-1].values()])
        n_optimistic = sum([tr_type == 'Optimistic' for tr_type in info.sentiments[-1].values()])
        n_pessimists = sum([tr_type == 'Pessimistic' for tr_type in info.sentiments[-1].values()])

        dp = info.prices[-1] - info.prices[-2] if len(info.prices) > 1 else 0  # price derivative
        if self.sentiment == 'Optimistic':
            p = min([self.markets[_].price() for _ in range(len(self.markets))])  # market price
            x = (n_optimistic - n_pessimists) / n_chartists

            U = a1 * x + a2 / v1 * dp / p
            prob = v1 * n_chartists / n_traders * exp(U)
            if prob > random.random():
                self.sentiment = 'Pessimistic'

        elif self.sentiment == 'Pessimistic':
            p = max([self.markets[_].price() for _ in range(len(self.markets))])  # market price
            x = (n_optimistic - n_pessimists) / n_chartists

            U = a1 * x + a2 / v1 * dp / p
            prob = v1 * n_chartists / n_traders * exp(-U)
            if prob > random.random():
                self.sentiment = 'Optimistic'

    def refresh(self, info):
        self.change_sentiment(info)


class Universalist(Fundamentalist, Chartist):
    """
    Universalist mixes Fundamentalist, Chartist trading strategies allowing to change one strategy to another
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None, access: int = 1):
        """
        :param markets: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Chartist' if random.random() > .5 else 'Fundamentalist'  # randomly decide type
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'  # sentiment about trend (Chartist)
        self.access = access  # next n dividend payments known (Fundamentalist)

    def call(self):
        """
        Call one of parents' methods depending on what type it is currently set.
        """
        if self.type == 'Chartist':
            Chartist.call(self)
        elif self.type == 'Fundamentalist':
            Fundamentalist.call(self)

    def change_strategy(self, info, a1=1, a2=1, a3=1, v1=.1, v2=.1, s=.1):
        """
        Change strategy or sentiment

        :param info: SimulatorInfo
        :param a1: importance of chartists opinion
        :param a2: importance of current price changes
        :param a3: importance of fundamentalist profit
        :param v1: frequency of revaluation of opinion for sentiment
        :param v2: frequency of revaluation of opinion for strategy
        :param s: importance of fundamental value opportunities
        """
        # Gather variables
        n_traders = len(info.traders)  # number of all traders
        n_fundamentalists = sum([tr.type == 'Fundamentalist' for tr in info.traders.values()])
        n_optimistic = sum([tr.sentiment == 'Optimistic' for tr in info.traders.values() if tr.type == 'Chartist'])
        n_pessimists = sum([tr.sentiment == 'Pessimistic' for tr in info.traders.values() if tr.type == 'Chartist'])

        dp = info.prices[-1] - info.prices[-2] if len(info.prices) > 1 else 0  # price derivative
        p = self.markets[0].price()  # market price
        pf = self.evaluate(self.markets[0].dividend(self.access), self.markets[0].risk_free())  # fundamental price
        r = pf * self.markets[0].risk_free()  # expected dividend return
        R = mean(info.returns[-1].values())  # average return in economy

        # Change sentiment
        if self.type == 'Chartist':
            Chartist.change_sentiment(self, info, a1, a2, v1)

        # Change strategy
        U1 = max(-100, min(100, a3 * ((r + 1 / v2 * dp) / p - R - s * abs((pf - p) / p))))
        U2 = max(-100, min(100, a3 * (R - (r + 1 / v2 * dp) / p - s * abs((pf - p) / p))))

        if self.type == 'Chartist':
            if self.sentiment == 'Optimistic':
                prob = v2 * n_optimistic / (n_traders * exp(U1))
                if prob > random.random():
                    self.type = 'Fundamentalist'
            elif self.sentiment == 'Pessimistic':
                prob = v2 * n_pessimists / (n_traders * exp(U2))
                if prob > random.random():
                    self.type = 'Fundamentalist'

        elif self.type == 'Fundamentalist':
            prob = v2 * n_fundamentalists / (n_traders * exp(-U1))
            if prob > random.random() and self.sentiment == 'Pessimistic':
                self.type = 'Chartist'
                self.sentiment = 'Optimistic'

            prob = v2 * n_fundamentalists / (n_traders * exp(-U2))
            if prob > random.random() and self.sentiment == 'Optimistic':
                self.type = 'Chartist'
                self.sentiment = 'Pessimistic'

    def refresh(self, info):
        self.change_strategy(info)


class MarketMaker(Trader):
    """
    MarketMaker creates limit orders on both sides of the spread trying to gain on
    spread between bid and ask prices, and maintain its assets to cash ratio in balance.
    """

    def __init__(self, markets: List[Broker], cash: float, assets: List[int] = None, softlimits: List[int] = None):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        if softlimits is None:
            softlimits = [100] * len(self.markets)
        self.softlimits = softlimits
        self.type = 'Market Maker'
        self.uls = self.softlimits
        self.lls = [-softlimit for softlimit in self.softlimits]
        self.panic = False
        self.prev_cash = cash

    def call(self):
        logging.Logger.info(f"Market Maker {self.id} PnL {self.cash - self.prev_cash}. Cash: {self.cash}")
        # Clear previous orders
        for order in self.orders.copy():
            self._cancel_order(order)

        # Calculate total bid and ask volume for all markets
        total_bid_volume = 0
        total_ask_volume = 0
        for i in range(len(self.markets)):
            bid_volume = max(0, (self.uls[i] - 1 - self.assets[i]) // 2)
            ask_volume = max(0, (self.assets[i] - 1 - self.uls[i]) // 2)
            total_bid_volume += bid_volume
            total_ask_volume += ask_volume

        # If in panic state we only either sell or buy commodities
        if not total_bid_volume or not total_ask_volume:
            self.panic = True
            self._buy_market(
                sum(self.lls[i] + self.lls[i] for i in range(len(self.markets)))) if total_ask_volume == 0 else None
            self._sell_market((sum(self.assets))) if total_bid_volume == 0 else None
        else:
            # Calculate spread and price offset for each market
            for i in range(len(self.markets)):
                spread = self.markets[i].spread()
                base_offset = min(1, (spread['ask'] - spread['bid']) * (self.assets[i] / self.lls[i]))
                bid_volume = max(0, (self.uls[i] - 1 - self.assets[i]) // 2)
                ask_volume = max(0, (self.assets[i] - 1 - self.lls[i]) // 2)
                bid_price = spread['bid'] + base_offset
                ask_price = spread['ask'] - base_offset
                self._buy_limit(bid_volume, bid_price, i)
                self._sell_limit(ask_volume, ask_price, i)
            self.panic = False
        self.prev_cash = self.cash
    # def call(self):
    #     logging.LOGGER.info("PnL", self.cash - self.prev_cash, f"{self.cash}")
    #     # Clear previous orders
    #     for order in self.orders.copy():
    #         self._cancel_order(order)
    #
    #     # Calculate total bid and ask volume for all markets
    #     total_bid_volume = 0
    #     total_ask_volume = 0
    #     for i in range(len(self.markets)):
    #         bid_volume = max(0, (self.uls[i] - 1 - self.assets[i]) // 2)
    #         ask_volume = max(0, (self.assets[i] - 1 - self.lls[i]) // 2)
    #         total_bid_volume += bid_volume
    #         total_ask_volume += ask_volume
    #
    #     # If in panic state we only either sell or buy commodities
    #     if not total_bid_volume or not total_ask_volume:
    #         self.panic = True
    #         if total_ask_volume == 0:
    #             # Sell all assets
    #             sell_volume = sum(self.assets)
    #             self._sell_market(sell_volume)
    #         if total_bid_volume == 0:
    #             # Buy as much as possible based on soft limits
    #             buy_volume = sum(self.uls) + sum(self.lls)
    #             self._buy_market(buy_volume)
    #     else:
    #         # Calculate spread and price offset for each market
    #         for i in range(len(self.markets)):
    #             spread = self.markets[i].spread()
    #             base_offset = min(1, (spread['ask'] - spread['bid']) * (self.assets[i] / self.lls[i]))
    #             bid_volume = max(0, (self.uls[i] - 1 - self.assets[i]) // 2)
    #             ask_volume = max(0, (self.assets[i] - 1 - self.lls[i]) // 2)
    #             bid_price = spread['bid'] + base_offset
    #             ask_price = spread['ask'] - base_offset
    #             self._buy_limit(bid_volume, bid_price, i)
    #             self._sell_limit(ask_volume, ask_price, i)
    #         self.panic = False
    #     self.prev_cash = self.cash


class AwareTrader(Trader):
    def __init__(self, hesitation: float, delay: int, markets: List[Broker], cash: float or int,
                 assets: List[int] = None):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.info_flow = InfoFlow()
        # this is reciprocal of the share of cash/assets,
        # that the agent is willing to offer at a time
        self.hesitation = hesitation
        self.delay = delay
        self.market = markets[0]

    def inform(self, news):
        self.info_flow.put(self.delay, news)


class NumericalFundamentalist(AwareTrader):
    def __init__(self, expectation: float, delay: int, markets: List[Broker], cash: float or int,
                 assets: List[int] = None):
        super().__init__(6.0, delay, markets, cash, assets if assets is not None else [0] * len(markets))
        self.expectation = expectation

    def call(self):
        price = self.market.price()
        if price is None:
            return
        news = self.info_flow.pull()
        if type(news) is CategoricalNews:
            return
        if type(news) is NumericalNews:
            if news.performance > self.expectation:
                self._sell_limit(sum(self.assets) // self.hesitation, price, market_id=0)
            else:
                q = round(self.cash / self.hesitation / price)
                if q > 0:
                    self._buy_limit(q, price, market_id=0)


class AdaptiveNumericalFundamentalist(AwareTrader):
    def __init__(self, phi: float, expectation: float, delay: int, markets: List[Broker], cash: float or int,
                 assets: List[int] = None):
        super().__init__(6.0, delay, markets, cash, assets if assets is not None else [0] * len(markets))
        self.expectation = expectation
        self.phi = phi

    @staticmethod
    def smooth(coef, old, new):
        return old * (1 - coef) + new * coef

    def call(self):
        price = self.market.price()
        if price is None:
            return
        news = self.info_flow.pull()
        if type(news) is not NumericalNews:
            return
        if news.performance > self.expectation:
            self._sell_limit(self.assets[0] // self.hesitation, price, 0)
        else:
            q = round(self.cash / self.hesitation / self.market.price())
            if q > 0:
                self._sell_limit(q, price, 0)
        self.expectation = AdaptiveNumericalFundamentalist.smooth(self.phi, self.expectation, news.performance)


class LossAverseTrader(Fundamentalist, Chartist):
    # https://www.hindawi.com/journals/ads/2015/971269/
    """
    Universalist mixes Fundamentalist, Chartist trading strategies allowing to change one strategy to another
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None, access: int = 1,
                 loss_aversion_parameter: float = 2.25, b: float = 0.04, c: float = 0.04, m: float = 0.975,
                 beta_std: float = 0.05, gamma_std: float = 0.01, r: float = 300):
        """
        :param markets: exchange agent links
        :param cash: number of cash
        :param assets: number of assets
        :param access: number of future dividends informed
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Chartist' if random.random() > .5 else 'Fundamentalist'  # randomly decide type
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'  # sentiment about trend (Chartist)
        self.access = access  # next n dividend payments known (Fundamentalist)
        self.loss_aversion_parameter = loss_aversion_parameter
        self.b = b  # Extrapolating parameter
        self.c = c  # Reverting parameter
        self.m = m  # Memory parameter
        self.r = r  # Intensity of choice parameter
        self.beta_std = beta_std  # Standard deviation of the additional random orders of technical trading
        self.gamma_std = gamma_std  # Standard deviation of the additional random orders of fundamental trading

        self.fundamental_orders = []
        self.chartist_orders = []
        self.prev_fundamental_attractiveness = 0
        self.prev_chartist_attractiveness = 0
        self.prev_price = self.markets[0].price()

    def call(self):
        """
        Call one of parents' methods depending on what type it is currently set.
        """
        if self.type == 'Chartist':
            Chartist.call(self)
        elif self.type == 'Fundamentalist':
            Fundamentalist.call(self)
        else:
            return

    def change_strategy(self, info):
        """
        Change strategy or sentiment

        """
        # Gather variables
        if self.markets[0].iteration() < 2:
            beta, gamma = random.normalvariate(mu=0, sigma=self.beta_std), random.normalvariate(mu=0,
                                                                                                sigma=self.gamma_std)
            p = self.markets[0].price()  # market price
            pf = self.evaluate(self.markets[0].dividend(self.access), self.markets[0].risk_free())
            self.fundamental_orders.append(self.b * (p - self.prev_price) + beta)
            self.chartist_orders.append(self.c * (pf - p) + gamma)
            self.prev_price = self.markets[0].price()
            return

        beta, gamma = random.normalvariate(mu=0, sigma=self.beta_std), random.normalvariate(mu=0, sigma=self.gamma_std)

        p = self.markets[0].price()  # market price
        pf = self.evaluate(self.markets[0].dividend(self.access), self.markets[0].risk_free())  # fundamental price

        new_fundamental_order = self.b * (p - self.prev_price) + beta
        new_chartist_order = self.c * (pf - p) + gamma

        chartist_attractiveness = (p - self.prev_price) * self.chartist_orders[0] + \
                                  self.m * self.prev_chartist_attractiveness
        fundamental_attractiveness = (p - self.prev_price) * self.fundamental_orders[0] + \
                                     self.m * self.prev_fundamental_attractiveness

        vc = chartist_attractiveness * self.r
        vf = fundamental_attractiveness * self.r

        if self.type == 'Chartist' and vc < 0:
            vc *= self.loss_aversion_parameter
        elif self.type == 'Fundamentalist' and vf < 0:
            vf *= self.loss_aversion_parameter

        denominator = vc + vf + 1

        wc = vc / denominator
        wf = vf / denominator
        w0 = 1 / denominator

        if wc > wf and wc > w0:
            self.type = 'Chartist'
        elif wf > wc and wf > w0:
            self.type = 'Fundamentalist'
        else:
            self.type = 'Pass'

        self.prev_fundamental_attractiveness = fundamental_attractiveness
        self.prev_chartist_attractiveness = chartist_attractiveness
        self.prev_price = self.markets[0].price()
        self.fundamental_orders = [self.fundamental_orders[1], new_fundamental_order]
        self.chartist_orders = [self.chartist_orders[1], new_chartist_order]

    def refresh(self, info):
        self.change_strategy(info)


class LiquidityConsumer(Trader):
    # https://eprints.soton.ac.uk/423233/2/McGroarty2018_Article_HighFrequencyTradingStrategies.pdf
    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None,
                 min_volume: float = 1000, max_volume: float = 100000, acting_prob: float = 0.1):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Liquidity Consumer'
        self.order_volume = random.uniform(min_volume, max_volume)
        self.acting_prob = acting_prob
        if random.random() > 0.5:
            self.action = 'buy'
        else:
            self.action = 'sell'
        if self.acting_prob is None:
            self.acting_prob = random.uniform(0.05, 0.95)

    def call(self):
        random_state = random.random()
        if random_state < self.acting_prob:
            random_state = random.random()
            if self.action == 'buy':
                best_volume = min(self.markets[0].spread_volume()['bid'], self.order_volume)
                self._buy_market(best_volume)
            else:
                best_volume = min(self.markets[0].spread_volume()['ask'], self.order_volume)
                self._sell_market(best_volume)
            self.order_volume -= best_volume


class MomentumTrader(Trader):
    # https://eprints.soton.ac.uk/423233/2/McGroarty2018_Article_HighFrequencyTradingStrategies.pdf
    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None, lag: int = 5,
                 order_percent: float = 0.05, entry_threshold: float = 0.001, order_limit: float = 1000,
                 acting_prob: float = 0.4):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Momentum Trader'
        self.roc = 0
        self.lag = lag
        self.lag_price = Queue()
        for i in range(lag):
            self.lag_price.put(100.)
        self.entry_threshold = entry_threshold
        self.order_limit = order_limit
        self.acting_prob = acting_prob
        self.order_percent = order_percent

    def call(self):
        random_state = random.random()
        lag_price = self.lag_price.get()
        if random_state < self.acting_prob:
            self.roc = (self.markets[0].price() - lag_price) / lag_price
            print(self.cash * abs(self.roc) * self.order_percent)
            if self.roc >= self.entry_threshold:
                self._buy_market(self.cash * abs(self.roc) * self.order_percent)
            elif self.roc <= -self.entry_threshold:
                self._sell_market(self.cash * abs(self.roc) * self.order_percent)
        self.lag_price.put(self.markets[0].price())


class MeanReversionTrader(Trader):
    # https://eprints.soton.ac.uk/423233/2/McGroarty2018_Article_HighFrequencyTradingStrategies.pdf
    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None, k: int = 3,
                 sigma: float = 0.02,
                 discount_factor: float = 0.06, order_volume: float = 1, acting_prob: float = 0.4,
                 tick_size: float = 0.01):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Mean Reversion Trader'
        self.ema = self.markets[0].price()
        self.discount_factor = discount_factor
        self.order_volume = order_volume
        self.k = k
        self.sigma = sigma
        self.acting_prob = acting_prob
        self.tick_size = tick_size

    def call(self):
        p = self.markets[0].price()
        self.ema += self.discount_factor * (p - self.ema)
        if random.random() < self.acting_prob:
            spread = self.markets[0].spread()
            if p - self.ema >= self.k * self.sigma:
                self._sell_limit(self.order_volume, spread['ask'] - self.tick_size, 0)
            elif self.ema - p >= self.k * self.sigma:
                self._buy_limit(self.order_volume, spread['bid'] + self.tick_size, 0)


class AnchoringTrader(Chartist):
    """
    Fundamentalist evaluate stock value using Constant Dividend Model. Then places orders accordingly
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None,
                 discount_factor: float = 0.06, belief_weight: float = 0.1, shock_indicator: float = 10,
                 strategy_change_factor: float = 0.06,
    ):
        """
        :param markets: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Chartist'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'
        self.ema = self.markets[0].price()
        self.discount_factor = discount_factor
        self.belief_weight = belief_weight
        self.shock_indicator = shock_indicator
        self.strategy_change_factor = strategy_change_factor
        self.shock = False
        self.anchor = 0
        self.anchor_ema = 0

    def call(self):
        Chartist.call(self)

    def change_strategy(self, info):
        p = self.markets[0].price()
        if abs(p - self.ema) > self.shock_indicator:
            self.shock = True
            self.ema = self.markets[0].price()
        else:
            self.ema += self.discount_factor * (p - self.ema)
        if not self.shock:
            self.anchor_ema = self.anchor_ema
        else:
            self.anchor_ema += self.belief_weight * (self.ema - self.anchor_ema)
        if p - self.anchor_ema >= self.strategy_change_factor:
            self.sentiment = 'Optimistic'
        elif self.ema - p >= self.strategy_change_factor:
            self.sentiment = 'Pessimistic'

    def refresh(self, info):
        self.change_strategy(info)


class LowFrequencyTrader(Trader):
    #  https://informs-sim.org/wsc15papers/027.pdf
    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None,
                 max_memory_length: int = 30, order_life: int = 10, evolution_circle: int = 30, evolution_rate=0.3,
                 std1: float = 0.3, std2: float = 0.6, std3: float = 0.1, min_size_fluc: float = 2,
                 max_size_fluc: float = 10):
        """
        :param markets: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Low Frequency Trader'
        self.activation_probability = random.uniform(0.1, 0.9)
        self.max_memory_length = max_memory_length
        self.memory_length = random.randint(1, max_memory_length)
        self.order_life = order_life
        self.evolution_circle = evolution_circle
        self.evolution_rate = evolution_rate
        self.prev_price = [self.markets[0].price()]
        self.n1 = random.normalvariate(0, std1)
        self.n2 = random.normalvariate(0, std2)
        self.n3 = random.normalvariate(0, std3)
        self.std1 = std1
        self.std2 = std2
        self.std3 = std3
        self.min_size_fluc = min_size_fluc
        self.max_size_fluc = max_size_fluc
        self.size_fluctuation = random.uniform(min_size_fluc, max_size_fluc)
        self.price_fluctuation = random.uniform(-0.002, 0.01)
        self.prev_equity = self.equity()

    def call(self):
        for order in self.orders.copy():
            if self.markets[0].iteration() - order.iteration >= self.order_life:
                self._cancel_order(order)

        p = self.markets[0].price()
        if self.markets[0].iteration() > 0 and self.markets[0].iteration() % self.evolution_circle == 0:
            if self.equity() > self.prev_equity:
                self.activation_probability = random.uniform(self.activation_probability, 0.9)
                self.size_fluctuation = random.uniform(self.size_fluctuation, self.max_size_fluc)
            else:
                if random.uniform(0, 1) < self.evolution_rate:
                    self.n1 = random.normalvariate(0, self.std1)
                    self.n2 = random.normalvariate(0, self.std2)
                    self.n3 = random.normalvariate(0, self.std3)
                    self.memory_length = random.randint(1, self.max_memory_length)
            self.prev_equity = self.equity()
        if random.random() < self.activation_probability:
            self.prev_price.append(p)
            if len(self.prev_price) > self.memory_length:
                self.prev_price = self.prev_price[(len(self.prev_price) - self.memory_length):]
            return
        pf = Fundamentalist.evaluate(self.markets[0].dividend(1), self.markets[0].risk_free())

        expected_return = self.n1 * log(pf / p) + self.n3 * random.normalvariate(0, 1)
        if len(self.prev_price) > 1:
            expected_return += self.n2 * sum([self.prev_price[-1] / self.prev_price[-i]
                                              for i in range(2, len(self.prev_price) + 1)]) / len(self.prev_price)
        expected_price = p * exp(expected_return)
        ask_price = self.prev_price[-1] * (1 - self.price_fluctuation)
        bid_price = self.prev_price[-1] * (1 + self.price_fluctuation)

        order_volume = abs(expected_return * self.size_fluctuation)
        spread = self.markets[0].spread()

        if ask_price > expected_price:
            self._sell_limit(order_volume, min(ask_price, spread['ask']), 0)  # ASK

        if bid_price < expected_price:
            self._buy_limit(order_volume, max(bid_price, spread['bid']), 0)  # BID

        self.prev_price.append(p)
        if len(self.prev_price) > self.memory_length:
            self.prev_price = self.prev_price[(len(self.prev_price) - self.memory_length):]


class IntradayTrader(Trader):
    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None,
                 day_len: int = 12, night_len: int = 8, **kwargs):
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets), **kwargs)
        self.day_len = day_len
        self.night_len = night_len
        self._prev_day = False

    def assets_to_cash(self):
        for i in range(len(self.markets)):
            if self.assets[i] == 0:
                continue
            if self.assets[i] < 0:
                if self.markets[i].order_book()['ask'].last is None:
                    continue
                order = Order(
                    self.markets[i].order_book()['ask'].last.price,
                    -self.assets[i], 'bid', i,
                    self.markets[i].iteration(), self
                )
            else:
                if self.markets[i].order_book()['bid'].last is None:
                    continue
                order = Order(
                    self.markets[i].order_book()['bid'].last.price,
                    self.assets[i], 'ask', i,
                    self.markets[i].iteration(), self
                )
            self.markets[i].market_order(order)

    def call(self):
        is_day = self.markets[0].iteration() % (self.day_len + self.night_len) < self.day_len
        if is_day:
            super().call()
        elif self._prev_day:
            for order in self.orders.copy():
                self._cancel_order(order)
            self.assets_to_cash()
        self._prev_day = is_day

    def refresh(self, info):
        super().refresh(info)


class TrailingInfo:
    """
    Just a light-weight wrapper for information to reduce TrailingAgents' memory usage
    """
    def __init__(self, traders):
        chartists = [t for t in traders if t.type == 'Chartist']
        self.n_traders = len(traders)
        self.n_chartists = len(chartists)
        self.n_optimistic = len([t for t in chartists if t.sentiment == 'Optimistic'])
        self.n_pessimistic = len(chartists) - self.n_optimistic


class TrailingAgent(Trader):
    """
    An agent that trails the actions of other agents with some level of noise, randomness
    and information lag.
    """

    def __init__(self, markets: List[Broker], cash: float or int, assets: List[int] = None,
                 target_type: str = 'Chartist', info_lag: int = 1, trailing_factor: float = 0.9,
                 qty_limit: float = .1):
        """
        :param markets: exchange agent link
        :param cash: number of cash
        :param assets: number of assets
        :param target_type: type of agents we will trail on
        :param info_lag: number of iterations before agent gets info (must be >= 1)
        :param trailing_factor: how strictly should it trail the orders
        :param qty_limit: max fraction of orders cost to agent's inventory (0 to turn off)
        """
        super().__init__(markets, cash, assets if assets is not None else [0] * len(markets))
        self.type = 'Trailing Agent'
        self.sentiment = 'Optimistic' if random.random() > .5 else 'Pessimistic'

        self.target_type = target_type
        self.trailing_factor = trailing_factor
        self.info_lag = info_lag
        self.qty_limit = qty_limit

        self._info_queue = queue.Queue(maxsize=self.info_lag)
        self._order_queue = queue.Queue(maxsize=self.info_lag)
        self._target_traders = []
        self._iter_info = False  # did we get market info on this iteration (for multi-market simulations)

    def refresh(self, info):
        """
        We assume that TrailingAgent is not aware of the newest changes
        so we will just cache it in "inaccessible" part of memory
        """
        if not self._target_traders:
            self._target_traders = [
                trader for trader in info.traders.values() if trader.type == self.target_type
            ]
        if not self._iter_info:
            info = TrailingInfo(info.traders.values())
            self._info_queue.put_nowait(info)

            if self._info_queue.full():
                delayed_info = self._info_queue.get_nowait()
                self.change_sentiment(delayed_info)
            self._iter_info = True

    def change_sentiment(self, info: TrailingInfo, freq: float = 0.15):
        """
        :param info: SimulatorInfo
        :param freq: frequency of revaluation of opinion for sentiment
        """
        x = (info.n_optimistic - info.n_pessimistic) / info.n_chartists
        if self.sentiment == 'Pessimistic':
            x *= -1

        prob = freq * info.n_chartists / info.n_traders * exp(x)
        if prob > random.random():
            self.sentiment = 'Pessimistic' if self.sentiment == 'Optimistic' else 'Optimistic'

    def call(self):
        self._iter_info = False
        iteration = self.markets[0].iteration()
        order_type = 'bid' if self.sentiment == 'Optimistic' else 'ask'

        orders: list[Order] = []
        for trader in self._target_traders:
            for order in trader.orders:
                if order.iteration == iteration - self.info_lag and order.order_type == order_type:
                    orders.append(order)
        if not orders:
            return

        price = random.choice([order.price for order in orders])
        qty = Random.draw_quantity()
        if abs(self.qty_limit) > 1e-9:
            qty = min(qty, self.qty_limit * self.equity())
        qty = round(qty)
        if qty < 1:
            return

        if random.random() < self.trailing_factor:
            if order_type == 'bid':
                self._buy_limit(qty, price, 0)
            else:
                self._sell_limit(qty, price, 0)
