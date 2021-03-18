import hypernetx as hnx
import networkx as nx
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Notebooks import smalltest
from Notebooks import middletest

def hyperuse():
    Edges, Names, Weights = pickle.load(open("../Data/GoT.pkl", "rb"))
    # HG = hnx.Hypergraph(dict(enumerate(Edges)))  198node,1400+edges
    smallEdges = []
    for i in range(30):
        smallEdges.append(Edges[i])
    smallHG = hnx.Hypergraph(dict(enumerate(smallEdges)))
    smalltest.point_intensity_centrality(smallHG, 7)
    # smalltest.node_clustering_coefficient(smallHG, 7)
    # edges.element的形式 0: Entity(0,['0', '2'],{})
    #print(HG.edge_adjacency_matrix()) #边的邻接矩阵
    # print(HG.degree(str(HG.edges.elements.get(0).uid)))
    # print(HG.degree('0'))

#198node 1491edge
    # hnx.draw(HG)
    # plt.show()
    # g = nx.karate_club_graph()

def hypertest(H:hnx.Hypergraph):
    smalltest.hyperdegree(H)


if __name__ == '__main__':
    hyperuse()


