from genome import Genome
from node_gene import NodeGene
from connection_gene import ConnectionGene
import unittest
import networkx as nx
import matplotlib.pyplot as plt
import logging, sys
import os
from draw_genome import draw_genome

class TestAddNodeMutation(unittest.TestCase):

    _TEST_OUTPUT_DIR = "tests/" + __name__ + "/"

    def test_add_node_mutation(self):
        log = logging.getLogger('test')
        if not os.path.exists(TestAddNodeMutation._TEST_OUTPUT_DIR):
            os.makedirs(TestAddNodeMutation._TEST_OUTPUT_DIR) 
        parent_x = Genome()

        #add parent_x nodes
        parent_x.node_genes = {}
        parent_x.node_genes[0] = NodeGene(NodeGene.INPUT, 0)
        parent_x.node_genes[1] = NodeGene(NodeGene.INPUT, 1)
        parent_x.node_genes[2] = NodeGene(NodeGene.OUTPUT, 2)

        #add connections
        c1= ConnectionGene(parent_x.node_genes[0],parent_x.node_genes[2])
        c2= ConnectionGene(parent_x.node_genes[1],parent_x.node_genes[2])
        parent_x.connection_genes[c1.INNOV_NUM] = c1
        parent_x.connection_genes[c2.INNOV_NUM] = c2

        draw_genome(parent_x, TestAddNodeMutation._TEST_OUTPUT_DIR+"before_mutation.png")
        parent_x.add_node_mutation()
        draw_genome(parent_x, TestAddNodeMutation._TEST_OUTPUT_DIR+"after_mutation.png")   

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)