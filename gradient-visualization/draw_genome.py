# Experiment only - not used in gradient viz
import matplotlib.pyplot as plt
import networkx as nx
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
import logging
log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
def draw_genome(ip_layer, op_layer, filename='sample'):

    G_output = nx.DiGraph()
    plt.clf()
    for node in ip_layer:
        G_output.add_node(node._id, state=node.STATE, layer='i')
    for node in op_layer:
        G_output.add_node(node._id, state=node.STATE, layer='o')

    for node in ip_layer:
        for connection in node.CONNECTIONS:
            G_output.add_edge(connection.IN_NODE._id, connection.OUT_NODE._id, weight=round(connection.weight,2))
    
    pos= nx.spring_layout(G_output)
    ed_labels = nx.get_edge_attributes(G_output,'weight')
    node_labels = nx.get_node_attributes(G_output, 'state')
    nx.draw(G_output, pos, font_weight='bold')
    nx.draw_networkx_nodes(G_output, pos, node_color='r', nodelist=[node._id for node in ip_layer])
    nx.draw_networkx_nodes(G_output, pos, node_color='y', nodelist=[node._id for node in op_layer])
    nx.draw_networkx_labels(G_output, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G_output, pos, edge_labels=ed_labels)
    plt.savefig(dir_path+"/"+filename)
    # plt.show()