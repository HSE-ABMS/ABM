
class Message:
    def __init__(self, kind):
        pass

class PriceChangeMessage(Message):
    def __init__(self, dir):
        self.dir = dir

class LiquidityShockMessage(Message):
    def __init__(self):
        pass

class InformationMessage(Message):
    def __init__(self):
        pass
