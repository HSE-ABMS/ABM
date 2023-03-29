class News:
    pass

class CategoricalNews(News):
    pass

class NumericalNews(News):
    def __init__(self, performance):
        self.performance = performance

class InfoFlow:
    def __init__(self):
        self.q = []

    def pull(self) -> News or None: 
        if len(self.q) == 0:
            return None
        p = self.q[0]
        if isinstance(p, News):
            self.q.remove(p)
            return p
        if p == 0:
            self.q.remove(p)
            return None
        self.q[0] = p - 1
        return None

    def put(self, delay: int, news: News):
        self.q.append(news)
        self.q.append(delay)

