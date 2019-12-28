import math
import random
import numpy


class MatcherNetwork:
    SIZE = 10
    ALPHA = 50

    def __init__(self, ranks, A, B, C, D, sigma):
        self.ranks = ranks
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.sigma = sigma
        #self.neurons = numpy.random.random((self.SIZE, self.SIZE, ))
        self.neurons = numpy.zeros((self.SIZE, self.SIZE, ))

    def response_function(self, u: float) -> float:
        return (1/2) * (1 + numpy.tanh(self.ALPHA * u))

    def _next_value(self, i, j) -> float:
        row_sum = numpy.sum(self.neurons, axis=1)
        col_sum = numpy.sum(self.neurons, axis=0)
        all_sum = numpy.sum(row_sum)
        cell_value = self.neurons[i][j]
        return (
            -(self.A * (row_sum[i] - cell_value))
            - (self.B * (col_sum[j] - cell_value))
            - (self.C * (all_sum - (self.SIZE + self.sigma)))
            - (self.D * self.ranks[i][j])
        )

    def _step(self, i, j) -> bool:
        next_value = self.response_function(self._next_value(i, j))
        if self.neurons[i][j] != next_value:
            self.neurons[i][j] = next_value
            return True

        return False

    def step(self) -> bool:
        i = math.floor(random.random() * self.SIZE)
        j = math.floor(random.random() * self.SIZE)
        return self._step(i, j)

    def epoch(self) -> bool:
        changed = False
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                changed |= self._step(i, j)

        return changed

    def random_epoch(self) -> bool:
        changed = False
        rows = list(range(self.SIZE))
        cols = list(range(self.SIZE))
        random.shuffle(rows)
        random.shuffle(cols)
        for i in rows:
            for j in cols:
                changed |= self._step(i, j)

        return changed

