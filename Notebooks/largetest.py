import hypernetx as hnx
import math
from Notebooks import smalltest
from Notebooks import middletest
import numpy as np
import scipy as sp

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


'''
香农熵
拉普拉斯矩阵 = 度矩阵（对角对称） - 邻接矩阵
'''


def shannon_entropy(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    degree_matrix = np.zeros(shape=(node_num, node_num))
    # 首先要求度矩阵
    for i in range(node_num):
        degree_matrix[i - 1, i - 1] = smalltest.hyperdegree(hg, i)
    # 创建拉普拉斯矩阵
    laplace_matrix = degree_matrix - matrix
    vals, vecs = sp.sparse.linalg.eigs(laplace_matrix)  # val特征值，vecs特征向量
    vals_sum = 0
    for i in range(len(vals)):
        vals_sum = vals_sum + math.fabs(vals[i]) * math.log2(math.fabs(vals[i]))
    print(-1 * vals_sum)


'''
网络效率
'''


def hyperNet_efficiency(hg: hnx.Hypergraph):
    # 暂缓，没想好节点之间的拓扑距离怎么算
    return


'''
网络的自然连通度
'''
def hyperNet_natural_connectivity(hg: hnx.Hypergraph):
    return