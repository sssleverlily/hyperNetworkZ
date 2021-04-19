import hypernetx as hnx
import networkx as nx
from Notebooks import micro_statistics
import matplotlib.pyplot as plt


def init():
    net = nx.read_gml('/Users/ssslever/PycharmProjects/hyperNetworkZ/Data/netscience.gml')
    edges = net.edges._adjdict
    smallEdges = {}
    i = 0
    for i, (k, v) in enumerate(edges.items()):
        smallEdges.update({k: v})
        if i == 30:
            print()
            break
    print(smallEdges)
    print(edges)
    # for i in range(30):
    #     smallEdges.append(edges.get(i))
    HG = hnx.Hypergraph(dict(enumerate(smallEdges)))
    # print(HG)
    hnx.draw(HG)
    plt.show()
    return


if __name__ == '__main__':
    init()