from genome import Genome
from node_gene import NodeGene
from connection_gene import ConnectionGene
import unittest
import networkx as nx
import matplotlib.pyplot as plt
import logging, sys

class TestBasicCrossover(unittest.TestCase):

    def test_crossover(self):
        log = logging.getLogger('test')
        parent_x = Genome()
        parent_y = Genome()

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

        G_x = nx.DiGraph()
        G_x.add_nodes_from([1,2,3,4,5])
        G_x.add_edges_from([(1,5),(1,4),(2,5),(3,4)])
        nx.draw(G_x, with_labels=True, font_weight='bold')
        plt.savefig("x.png")

        #add parent_y nodes
        parent_y.node_genes[1] = NodeGene(NodeGene.INPUT, 1)
        parent_y.node_genes[2] = NodeGene(NodeGene.INPUT, 2)
        parent_y.node_genes[3] = NodeGene(NodeGene.INPUT, 3)
        parent_y.node_genes[4] = NodeGene(NodeGene.OUTPUT, 4)
        parent_y.node_genes[5] = NodeGene(NodeGene.HIDDEN, 5)
        parent_y.node_genes[6] = NodeGene(NodeGene.HIDDEN, 6)

        #add connections
        parent_y.connection_genes[1] = ConnectionGene(parent_y.node_genes[1],parent_y.node_genes[6])
        parent_y.connection_genes[2] = ConnectionGene(parent_y.node_genes[1],parent_y.node_genes[4])
        parent_y.connection_genes[3] = ConnectionGene(parent_y.node_genes[2],parent_y.node_genes[5])
        parent_y.connection_genes[4] = ConnectionGene(parent_y.node_genes[3],parent_y.node_genes[4])
        parent_y.connection_genes[5] = ConnectionGene(parent_y.node_genes[3],parent_y.node_genes[5])
        parent_y.connection_genes[6] = ConnectionGene(parent_y.node_genes[5],parent_y.node_genes[6])
        parent_y.connection_genes[7] = ConnectionGene(parent_y.node_genes[6],parent_y.node_genes[4])

        plt.clf()
        G_y = nx.DiGraph()
        G_y.add_nodes_from([1,2,3,4,5,6])
        G_y.add_edges_from([(1,6),(1,4),(2,5),(3,4),(3,5),(5,6),(6,4)])
        nx.draw(G_y, with_labels=True, font_weight='bold')
        plt.savefig("y.png")

        # log.debug(parent_x.node_genes, parent_y.node_genes)
        child = Genome.crossover(parent_x, parent_y)

        plt.clf()
        G_c = nx.DiGraph()
        for _id, node in child.node_genes.items():
            G_c.add_node(node._id)
        for inv_num, connection in child.connection_genes.items():
            G_c.add_edge(connection.in_node, connection.out_node)
        nx.draw(G_y, with_labels=True, font_weight='bold')
        plt.savefig("child.png")

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)