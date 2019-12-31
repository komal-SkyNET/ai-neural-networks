import random as r
import copy

class ConnectionGene:
    _INNOVATION_COUNTER = -1

    def __init__(self, in_node, out_node, weight=None, INNOV_NUM=None, expressed=True):
        ConnectionGene._INNOVATION_COUNTER += 1
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight if weight else r.random()
        self.expressed = expressed
        self.INNOV_NUM = INNOV_NUM if INNOV_NUM else ConnectionGene._INNOVATION_COUNTER

    def disable(self):
        self.expressed = False

    def copy(self):
        return copy.deepcopy(self)