import random as r
from connection_gene import ConnectionGene
from node_gene import NodeGene

class Genome:
    _PROBABILITY_PERTURBED = 0.9

    def __init__(self):
        self.connection_genes = {}
        self.node_genes = {}

    def add_connection_mutation(self):
        reverse = False
        connection_exists = False
        node_a = self.node_genes[r.randint(0, len(self.node_genes))]
        node_b = self.node_genes[r.randint(0, len(self.node_genes))]

        if node_a.type == NodeGene.HIDDEN & node_b.type == NodeGene.INPUT:
            reverse = True
        elif node_a.type == NodeGene.OUTPUT & node_b.type == NodeGene.HIDDEN:
            reverse = True
        elif node_a.type == NodeGene.OUTPUT & node_b.type == NodeGene.INPUT:
            reverse = True
        
        for connection in self.connection_genes.values():
            if node_a._id == connection.in_node & node_b._id == connection.out_node:
                connection_exists = True
                break
            elif node_b._id == connection.in_node & node_a._id == connection.out_node:
                connection_exists: True
                break
        
        if not connection_exists:
            if not reverse:
                new_connection = ConnectionGene(node_a, node_b)
            else:
                new_connection = ConnectionGene(node_b, node_a)

        self.connection_genes[new_connection.INNOV_NUM] = new_connection


    def add_node_mutation(self):
        con = self.connection_genes[r.randint(0, len(self.connection_genes)-1)]
        node_in = con.in_node
        node_out = con.out_node
        con.disable()
        # create new node 
        node_new = NodeGene(NodeGene.HIDDEN, len(self.node_genes))

        #create new connections for new node
        connection_in_new = ConnectionGene(node_in, node_new)
        connection_new_out = ConnectionGene(node_new, node_out)

        #add to Genome store
        self.node_genes[node_new._id] = node_new
        self.connection_genes[connection_in_new.INNOV_NUM] = connection_in_new
        self.connection_genes[connection_new_out.INNOV_NUM] = connection_new_out

    ## assume parent_x fittest
    @staticmethod 
    def crossover(parent_x, parent_y):
        child = Genome()
        #all nodes from fittest parent added to child
        for _id, node in parent_x.node_genes.items():
            child.node_genes[_id] = node.copy()
        for x_innov_number, x_connection_gene in parent_x.connection_genes.items():
            if parent_y.connection_genes.get(x_innov_number):
                #matching genes
                if bool(r.getrandbits(1)):
                    child.connection_genes[x_innov_number] = x_connection_gene.copy()
                else:
                    #use y connection
                    child.connection_genes[x_innov_number] = parent_y.connection_genes[x_innov_number].copy()
            else:
                #disjoint excess genes
                child.connection_genes[x_innov_number] = x_connection_gene.copy()
        return child

    def mutation(self):
        for connections in self.connection_genes.items():
            if r.random() > Genome._PROBABILITY_PERTURBED:
                ##TODO: uniformly perturbed distribution?
                connections.weight = r.uniform(-2,2) * connections.weight
            else:
                connections.weight = r.uniform(-2,2)
