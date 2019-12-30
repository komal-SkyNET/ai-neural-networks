from enum import Enum
import copy

class NodeGene:
    HIDDEN=1
    INPUT=2
    OUTPUT=3

    def __init__(self, _type, _id):
        self._type = _type
        self._id = _id

    def copy(self):
        return copy.deepcopy(self)
    