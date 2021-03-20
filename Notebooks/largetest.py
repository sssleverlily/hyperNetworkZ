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
    matrix = hg.adjacency_matrix().asfptype()  # 上投矩阵以浮动或翻倍
    node_num = matrix.shape[0]
    vals, vecs = sp.sparse.linalg.eigs(matrix)  # val特征值，vecs特征向量
    sum = 0
    for i in range(len(vals)):
        sum = sum + math.exp(vals[i - 1])
    natural_connectivity = math.log(sum / node_num, math.e)
    print(natural_connectivity)


'''
空间中心性
程度中心性  中间中心性 特征向量中心性
平方相加再开根号
'''


def spatial_centrality(hg: hnx.Hypergraph):
    return


'''
子图向心性
A = UDU(T) 其中A是邻接矩阵
平均子图向心性就再除节点数
'''


def subgraph_centripetality(hg: hnx.Hypergraph):
    adjacency_matrix = hg.adjacency_matrix().todense()
    matrix = hg.adjacency_matrix().asfptype()  # 上投矩阵以浮动或翻倍
    node_num = matrix.shape[0]
    vals, vecs = sp.sparse.linalg.eigs(matrix)  # val特征值，vecs特征向量
    return


'''
网络的节点分类熵
第i类节点在整个网络类型节点中占据的比例pi
'''

def classification_entropy(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    node_type_list = []
    node_type_classification = 0
    #寻找第i类节点（这里以度为例子）
    for i in range(node_num):
        node_type = 0
        for j in range(node_num):
            if smalltest.hyperdegree(hg, j) == i:
                node_type = node_type + 1
        node_type_list.append(node_type/node_num)
    for k in range(node_num):
        if node_type_list[k] != 0:
            node_type_classification = node_type_classification + (-1) * node_type_list[k]*math.log2(node_type_list[k])
    print(node_type_classification)