import matplotlib.pyplot as plt
import networkx as nx

def draw_genome(parent_x, abs_filename):
    G_output = nx.DiGraph()
    plt.clf()
    for _id in parent_x.node_genes.keys():
        G_output.add_node(_id)
    for connection in parent_x.connection_genes.values():
        if connection.expressed:
            G_output.add_edge(connection.in_node._id, connection.out_node._id)
    nx.draw(G_output, with_labels=True, font_weight='bold')
    plt.savefig(abs_filename)