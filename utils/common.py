

class OnlineAvg:

    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, new_x):
        self.n += 1
        self.avg = (self.avg * (self.n - 1) + new_x) / self.n
