from AgentBasedModel import Random, Simulator
from AgentBasedModel import Broker
from random import seed, random

class FakeBroker(Broker):
    def _not_impl(_):
        raise Exception("Not implemented")

    def __init__(self, price: float or int, spread: dict or None, ):
        self.fake_price = price
        self.fake_spread = spread
        self.limit_orders = []
        self.market_orders = []
        self.cancel_orders = []

    def generate_dividend(self):
        self._not_impl()

    def spread(self) -> dict or None:
        return self.fake_spread

    def spread_volume(self) -> dict or None:
        self._not_impl()

    def price(self) -> float:
        return self.fake_price

    def dividend(self, access: int = None) -> list or float:
        return []

    def limit_order(self, order):
        self.limit_orders.append(order)

    def market_order(self, order):
        self.market_orders.append(order)

    def cancel_order(self, order):
        self.cancel_orders.append(order)

def test_random1():
    broker = FakeBroker(5, {'bid': 10.0, 'ask': 9.0})
    seed(42)
    traders = [Random(broker, 100, [10])]
    Simulator(exchanges=[broker], traders=traders).simulate(n_iter=10)
    assert len(broker.limit_orders) != 0
