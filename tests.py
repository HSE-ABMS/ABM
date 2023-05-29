from AgentBasedModel import Random, Simulator
from AgentBasedModel import Broker
from AgentBasedModel import Trader
from random import seed, random

class FakeBroker(Broker):
    def __init__(self, price: float or int, spread: dict or None, ):
        self.fake_price = price
        self.fake_spread = spread
        self.limit_orders = []
        self.market_orders = []
        self.cancel_orders = []

        self._id = 0
        self._risk_free = 5e-4
        self._transaction_cost = 0.0

    def generate_dividend(self):
        # self._not_impl()
        pass

    def spread(self) -> dict or None:
        return self.fake_spread

    def spread_volume(self) -> dict or None:
        self._not_impl()

    def price(self) -> float:
        return self.fake_price

    def dividend(self, access: int = None) -> list or float:
        return 0.0

    def limit_order(self, order):
        self.limit_orders.append(order)

    def market_order(self, order):
        self.market_orders.append(order)

    def cancel_order(self, order):
        self.cancel_orders.append(order)

    def order_book(self):
        return {'bid': [], 'ask': []}

    def risk_free(self):
        return self._risk_free

    def iteration(self):
        return self._iteration

    def increment_iteration(self):
        self._iteration += 1

    def transaction_cost(self):
        return self._transaction_cost

    def id(self):
        return self._id

class FakeTrader(Trader):
    def __init__(self, markets, cash, assets):
        super().__init__(markets, cash, assets)

    def call(self):
        self._buy_limit(1, 5, 0)

def test_faketrader_order_count_1():
    broker = FakeBroker(5, {'bid': 10.0, 'ask': 9.0})
    seed(42)
    traders = [FakeTrader([broker], 100, [10])]
    Simulator(exchanges=[broker], traders=traders).simulate(n_iter=10)
    assert len(broker.limit_orders) == 10
    assert len(broker.market_orders) == 0

def test_faketrader_order_count_2():
    broker = FakeBroker(5, {'bid': 10.0, 'ask': 9.0})
    seed(42)
    traders = [FakeTrader([broker], 100, [10]) for _ in range(20)]
    Simulator(exchanges=[broker], traders=traders).simulate(n_iter=10)
    assert len(broker.limit_orders) == 200
    assert len(broker.market_orders) == 0