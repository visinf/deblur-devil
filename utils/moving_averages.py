# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

class MovingAverage:
    postfix = "avg"

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add_value(self, sigma, addcount=1):
        self.sum += sigma
        self.count += addcount

    def add_average(self, avg, addcount):
        self.sum += avg * addcount
        self.count += addcount

    def mean(self):
        return self.sum / self.count


class ExponentialMovingAverage:
    postfix = "ema"

    def __init__(self, alpha=0.7):
        self.weighted_sum = 0.0
        self.weighted_count = 0
        self.alpha = alpha

    def add_value(self, sigma):
        self.weighted_sum = sigma + (1.0 - self.alpha) * self.weighted_sum
        self.weighted_count = 1 + (1.0 - self.alpha) * self.weighted_count

    def add_average(self, avg, addcount):
        self.weighted_sum = avg * addcount + (1.0 - self.alpha) * self.weighted_sum
        self.weighted_count = addcount + (1.0 - self.alpha) * self.weighted_count

    def mean(self):
        return self.weighted_sum / self.weighted_count
