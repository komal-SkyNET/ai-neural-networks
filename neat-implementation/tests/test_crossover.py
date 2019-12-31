from genome import Genome
from node_gene import NodeGene
from connection_gene import ConnectionGene
import unittest
import networkx as nx
import matplotlib.pyplot as plt
import logging, sys, os
from draw_genome import draw_genome

class TestBasicCrossover(unittest.TestCase):

    _TEST_OUTPUT_DIR = "tests/" + __name__ + "/"

    def test_crossover(self):
        log = logging.getLogger('test')
        parent_x = Genome()
        parent_y = Genome()
        if not os.path.exists(TestBasicCrossover._TEST_OUTPUT_DIR):
            os.makedirs(TestBasicCrossover._TEST_OUTPUT_DIR) 
        
        #add parent_x nodes
        parent_x.node_genes = {}
        parent_x.node_genes[1] = NodeGene(NodeGene.INPUT, 1)
        parent_x.node_genes[2] = NodeGene(NodeGene.INPUT, 2)
        parent_x.node_genes[3] = NodeGene(NodeGene.INPUT, 3)
        parent_x.node_genes[4] = NodeGene(NodeGene.OUTPUT, 4)
        parent_x.node_genes[5] = NodeGene(NodeGene.HIDDEN, 5)

        #add connections
        parent_x.connection_genes[1] = ConnectionGene(parent_x.node_genes[1],parent_x.node_genes[5])
        parent_x.connection_genes[2] = ConnectionGene(parent_x.node_genes[1],parent_x.node_genes[4])
        parent_x.connection_genes[3] = ConnectionGene(parent_x.node_genes[2],parent_x.node_genes[5])
        parent_x.connection_genes[4] = ConnectionGene(parent_x.node_genes[3],parent_x.node_genes[4])

        draw_genome(parent_x, TestBasicCrossover._TEST_OUTPUT_DIR + "x.png")

        #add parent_y nodes
        parent_y.node_genes[1] = NodeGene(NodeGene.INPUT, 1)
        parent_y.node_genes[2] = NodeGene(NodeGene.INPUT, 2)
        parent_y.node_genes[3] = NodeGene(NodeGene.INPUT, 3)
        parent_y.node_genes[4] = NodeGene(NodeGene.OUTPUT, 4)
        parent_y.node_genes[5] = NodeGene(NodeGene.HIDDEN, 5)
        parent_y.node_genes[6] = NodeGene(NodeGene.HIDDEN, 6)

        #add connections
        parent_y.connection_genes[8] = ConnectionGene(parent_y.node_genes[1],parent_y.node_genes[6])
        parent_y.connection_genes[2] = ConnectionGene(parent_y.node_genes[1],parent_y.node_genes[4])
        parent_y.connection_genes[3] = ConnectionGene(parent_y.node_genes[2],parent_y.node_genes[5])
        parent_y.connection_genes[4] = ConnectionGene(parent_y.node_genes[3],parent_y.node_genes[4])
        parent_y.connection_genes[5] = ConnectionGene(parent_y.node_genes[3],parent_y.node_genes[5])
        parent_y.connection_genes[6] = ConnectionGene(parent_y.node_genes[5],parent_y.node_genes[6])
        parent_y.connection_genes[7] = ConnectionGene(parent_y.node_genes[6],parent_y.node_genes[4])

        draw_genome(parent_y, TestBasicCrossover._TEST_OUTPUT_DIR + "y.png")
        child = Genome.crossover(parent_x, parent_y)
        draw_genome(child, TestBasicCrossover._TEST_OUTPUT_DIR + "child.png")


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)