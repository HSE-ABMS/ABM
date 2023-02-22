class News:
    pass

class CategoricalNews(News):
    pass

class NumericalNews(News):
    def __init__(self, performance):
        self.performance = performance
