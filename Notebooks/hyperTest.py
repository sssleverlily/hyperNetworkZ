import hypernetx as hnx
import networkx as nx
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Notebooks import micro_statistics
from Notebooks import meso_statistics
from Notebooks import macro_statistics
from Notebooks import shortest_road
from Notebooks import hyperdraw
from Notebooks import ER_network
from Notebooks import test

def hyperuse():
    Edges, Names, Weights = pickle.load(open("../Data/GoT.pkl", "rb"))
    # print(Edges, Names, Weights)
    # HG = hnx.Hypergraph(dict(enumerate(Edges)))
    # print(dict(enumerate(Edges)))
    # ER_network.init_erNet()
    # 198node,1400+edges
    smallEdges = []
    for i in range(30):
        smallEdges.append(Edges[i])
    smallHG = hnx.Hypergraph(dict(enumerate(smallEdges)))
    # hyperdraw.hypergraphdraw(smallHG)
    # hyperdraw.draw_node_degree(smallHG)
    # edges.element的形式 0: Entity(0,['0', '2'],{})
    # print(HG.edge_adjacency_matrix()) #边的邻接矩阵
    # print(HG.degree(str(HG.edges.elements.get(0).uid)))
    # print(HG.degree('0'))
    ER_network.one_order(smallHG)


# 198node 1491edge
# hnx.draw(HG)
# plt.show()
# g = nx.karate_club_graph()

def hypertest(H: hnx.Hypergraph):
    micro_statistics.hyperdegree(H)


if __name__ == '__main__':
    hyperuse()
