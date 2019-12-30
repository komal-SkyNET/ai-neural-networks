import random as r
import copy

class ConnectionGene:
    _INNOVATION_COUNTER = 0

    def __init__(self, in_node, out_node):
        ConnectionGene._INNOVATION_COUNTER += 1
        self.in_node = in_node
        self.out_node = out_node
        self.weight = r.random()
        self.expressed = bool(r.getrandbits(1))
        self.INNOV_NUM = ConnectionGene._INNOVATION_COUNTER

    def disable(self):
        self.expressed = False

    def copy(self):
        return copy.deepcopy(self)