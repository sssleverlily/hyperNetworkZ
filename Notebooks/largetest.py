import hypernetx as hnx
import math
from Notebooks import smalltest
from Notebooks import middletest

'''
超图密度
超边数量/2^节点数量-1
'''


def hypergraph_density(hg: hnx.Hypergraph):
    matrix = hg.incidence_matrix().todense()
    node_num = matrix.shape[0]
    edge_num = matrix.shape[1]
    density = edge_num / (math.pow(2, node_num) - 1)
    print(density)


'''
网络的聚类系数
节点的聚类系数的平均值
'''


def net_clustering_coefficient(hg: hnx.Hypergraph):
    matrix = hg.incidence_matrix().todense()
    node_num = matrix.shape[0]
    node_clustering = 0
    for i in range(node_num):
        node_clustering = node_clustering + smalltest.node_clustering_coefficient(hg, i)
    net_clustering = node_clustering / node_num
    print(net_clustering)


'''
超网络的子图中心度
节点的子图中心度平均值
'''


def net_subgraph_centrality(hg: hnx.Hypergraph):
    matrix = hg.incidence_matrix().todense()
    node_num = matrix.shape[0]
    node_centrality = 0
    for i in range(node_num):
        node_centrality = node_centrality + smalltest.Subgraph_centrality(hg, i)
    net_centrality = node_centrality / node_num
    print(net_centrality)
