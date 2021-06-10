import hypernetx as hnx
import math
from Notebooks import micro_statistics
from Notebooks import meso_statistics
import numpy as np
import scipy as sp
from Notebooks import shortest_road

'''
超图密度
超边数量/2^节点数量-1
'''


def hypergraph_density(hg: hnx.Hypergraph):
    matrix = hg.incidence_matrix().todense()
    node_num = matrix.shape[0]
    edge_num = matrix.shape[1]
    density = edge_num / (math.pow(2, node_num) - 1)
    return density


'''
网络的聚类系数
节点的聚类系数的平均值
'''


def net_clustering_coefficient(hg: hnx.Hypergraph):
    matrix = hg.incidence_matrix().todense()
    node_num = matrix.shape[0]
    node_clustering = 0
    for i in range(node_num):
        node_clustering = node_clustering + micro_statistics.node_clustering_coefficient(hg, i)
        print(node_clustering)
    net_clustering = node_clustering / node_num
    return net_clustering


'''
超网络的子图中心度
节点的子图中心度平均值
'''


def net_subgraph_centrality(hg: hnx.Hypergraph):
    matrix = hg.incidence_matrix().todense()
    node_num = matrix.shape[0]
    node_centrality = 0
    for i in range(node_num):
        node_centrality = node_centrality + micro_statistics.Subgraph_centrality(hg, i)
    net_centrality = node_centrality / node_num
    return net_centrality


'''
平均最短路径
最短路径可以分为两种情况讨论：超边内任意两点间的最短路径为1，另外，不在同一条超边内的任意两个点间最多可以通过两条超边相连通，所以不在同一条超边内的任意两点间的最短的路径为2。
'''


def average_shortest_path(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    shortest_path_num = 0
    for i in range(node_num):
        for j in range(node_num):
            if j != i:
                if matrix[i, j] != 0:
                    shortest_path_num = shortest_path_num + 1
                else:
                    shortest_path_num = shortest_path_num + shortest_road.shortest(hg, i, j)

    average_shortest_path = (2 * shortest_path_num / 2) / (
                node_num * (node_num - 1))  # shortest_path_num除2的原因是i到j和j到i算了两遍
    return average_shortest_path


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
        degree_matrix[i - 1, i - 1] = micro_statistics.hyperdegree(hg, i)
    # 创建拉普拉斯矩阵
    laplace_matrix = degree_matrix - matrix
    vals, vecs = sp.sparse.linalg.eigs(laplace_matrix)  # val特征值，vecs特征向量
    vals_sum = 0
    for i in range(len(vals)):
        vals_sum = vals_sum + math.fabs(vals[i]) * math.log2(math.fabs(vals[i]))
    return vals_sum


'''
网络效率
最短路径可以分为两种情况讨论：超边内任意两点间的最短路径为1
'''


def hyperNet_efficiency(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    shortest_path_num = 0
    for i in range(node_num):
        for j in range(node_num):
            if j != i:
                if matrix[i, j] != 0:
                    shortest_path_num = shortest_path_num + 1
                elif shortest_road.shortest(hg, i, j) == 0:
                    shortest_path_num = shortest_path_num
                else:
                    shortest_path_num = shortest_path_num + 1/shortest_road.shortest(hg, i, j)
    hyperNet_efficiency = (shortest_path_num/2) / (node_num * (node_num - 1))
    return hyperNet_efficiency


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
    return natural_connectivity

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
    # 寻找第i类节点（这里以度为例子）
    for i in range(node_num):
        node_type = 0
        for j in range(node_num):
            if micro_statistics.hyperdegree(hg, j) == i:
                node_type = node_type + 1
        node_type_list.append(node_type / node_num)
    for k in range(node_num):
        if node_type_list[k] != 0:
            node_type_classification = node_type_classification + (-1) * node_type_list[k] * math.log2(
                node_type_list[k])
    return node_type_classification


'''
模体熵
pi代表了第i类节点在整个网络节点数种占的比例
hi^n(j)第i类节点为模核的n(j)-模体熵
'''


def hypernet_motif_entropy(hg: hnx.Hypergraph):
    # 先求pi好了
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    node_type_list = []
    p_i = 0
    for i in range(node_num):
        node_type = 0
        for j in range(node_num):
            if micro_statistics.hyperdegree(hg, j) == i:
                node_type = node_type + 1
        node_type_list.append(node_type / node_num)  # pi的列表
    for k in range(len(node_type_list)):
        p_i = p_i + node_type_list[k]
    # print(p_i)
    # n-模体熵
    return p_i


'''
超网络的中心度
'''


def hyperNet_centrality(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    centrality_list = []
    centrality_sum = 0
    for i in range(node_num):
        centrality_list.append(micro_statistics.Subgraph_centrality(hg, i))
    for j in range(node_num):
        centrality_sum = centrality_sum + (max(centrality_list) - centrality_list[j])
    centrality = centrality_sum / (node_num - 2)
    return centrality
