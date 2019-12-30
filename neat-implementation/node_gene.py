from enum import Enum

class NodeGene:
    HIDDEN=1
    INPUT=2
    OUTPUT=3

    def __init__(self, _type, _id):
        self._type = _type
        self._id = _id

    def copy(self):
        return NodeGene(self._id, self._type)
    