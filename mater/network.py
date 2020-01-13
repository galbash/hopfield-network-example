import math
import random
import numpy
from dataclasses import dataclass


@dataclass
class NetworkParams:
    """
    Parameters to tweak the network
    """

    A: int
    B: int
    C: int
    D: int
    sigma: float
    alpha: int = 50


class MatcherNetwork:
    """
    The mathcer network
    """

    SIZE = 10

    def __init__(self, ranks: numpy.ndarray, params: NetworkParams):
        """
        :param ranks: the rankings females gave to males
        :param params: params to control network behaviour
        """
        self.ranks = ranks
        self.params = params
        self.neurons = numpy.random.rand(self.SIZE, self.SIZE)

    def _in_to_out(self, u: float) -> float:
        """
        transforms neuron input potential to neuron output potential
        :param u: input
        :return: next value
        """
        return (1 / 2) * (1 + numpy.tanh(self.params.alpha * u))

    def _input_potential(self, i: int, j: int) -> float:
        """
        calculate neuron input potential based on index
        """
        row_sum = numpy.sum(self.neurons, axis=1)
        col_sum = numpy.sum(self.neurons, axis=0)
        all_sum = numpy.sum(row_sum)
        cell_value = self.neurons[i][j]
        result = (
            -(self.params.A * (row_sum[i] - cell_value))
            - (self.params.B * (col_sum[j] - cell_value))
            - (self.params.C * (all_sum - (self.SIZE + self.params.sigma)))
            - (self.params.D * self.ranks[i][j])
        )

        return result

    def _step(self, i: int, j: int) -> bool:
        """
        advances a neuron at a given index
        :return: True if the value changed, False otherwise
        """
        next_value = self._in_to_out(self._input_potential(i, j))
        if self.neurons[i][j] != next_value:
            self.neurons[i][j] = next_value
            return True

        return False

    def step(self) -> bool:
        """
        makes a single step on a random neuron in the network
        :return: True if the value changed, False otherwise
        """
        i = math.floor(random.random() * self.SIZE)
        j = math.floor(random.random() * self.SIZE)
        return self._step(i, j)

    def epoch(self) -> bool:
        """
        makes a step for all the neurons in the network, in an ordered fashion
        :return: True if any neuron value changed, False otherwise
        """
        changed = False
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                changed |= self._step(i, j)

        return changed

    def random_epoch(self) -> bool:
        """
        makes a step for all the neurons in the network, in a random fashion
        :return: True if any neuron value changed, False otherwise
        """
        changed = False
        rows = list(range(self.SIZE))
        cols = list(range(self.SIZE))
        random.shuffle(rows)
        random.shuffle(cols)
        for i in rows:
            for j in cols:
                changed |= self._step(i, j)

        return changed
