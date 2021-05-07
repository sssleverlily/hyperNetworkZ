import hypernetx as hnx
import networkx as nx
from Notebooks import micro_statistics
from Notebooks import macro_statistics
from Notebooks import meso_statistics
import matplotlib.pyplot as plt


def init():
    net = nx.read_gml('/Users/ssslever/PycharmProjects/hyperNetworkZ/Data/adjnoun.gml')
    edges = net.edges._adjdict
    edges_num = net.edges
    nodes = net._node
    smallEdges = {}
    i = 0
    for i, (k, v) in enumerate(edges.items()):
        smallEdges.update({k: v})
        if i == 30:
            print()
            break
    # for i in range(30):
    #     smallEdges.append(edges.get(i))
    print(edges)
    print(nodes)
    HG = hnx.Hypergraph(dict(enumerate(smallEdges)))
    # print(HG)
    # print(macro_statistics.hypergraph_density(HG))
    # print(micro_statistics.hyperdegree(HG, 5))
    # print(HG)
    hnx.draw(HG)
    plt.show()
    return


if __name__ == '__main__':
    init()