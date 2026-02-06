import cupy as np


class Dropout:
    def __init__(self, in_n, drop_rate=0):
        self.in_n = in_n
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, x, training=False):
        if not training or self.drop_rate == 0:
            return x

        self.mask = (np.random.random(size=x.shape) > self.drop_rate).astype(x.dtype)

        return x * self.mask / (1 - self.drop_rate)

    def backward(self, inner_d):
        return inner_d if self.drop_rate == 0 else inner_d * self.mask / (1 - self.drop_rate)
