import hypernetx as hnx
import math

'''
邻接矩阵 是点点
关联矩阵 是点边
'''
'''
超边度
超边度的话就遍历列
'''


def hyperedge_degree(hg: hnx.Hypergraph, edge: int):
    matrix = hg.incidence_matrix().todense()
    size = matrix.shape[1]
    edge_degree = 0
    for i in range(size):
        edge_degree = edge_degree + matrix[i, edge]
    return edge_degree


# 边k的超图分布就是 度为K的边占整个网络节点数的比例
def hyperdistribution(hg: hnx.Hypergraph, edge_degree_is_k: int):
    edges_Num = len(hg.edges)
    edge_Sum = 0
    for i in range(edges_Num):
        if hg.degree(str(hg.edges.elements.get(i).uid)) == edge_degree_is_k:
            edge_Sum = edge_Sum + 1
    distribution = edge_Sum / edges_Num
    return distribution


'''
好像不太对，这里应该是两点之间的距离
'''


def topological_distance(hg: hnx.Hypergraph, edge_i: int, edge_j: int):
    matrix = hg.adjacency_matrix().todense()
    return matrix[edge_i, edge_j]


'''
超边基数
超边包含的节点的个数
'''


def hyperedge_cardinality(hg: hnx.Hypergraph, edge: int):
    matrix = hg.incidence_matrix().todense()  # 点边
    node_num = matrix.shape[0]
    node_in_edge_num = 0
    for i in range(node_num):
        if matrix[i, edge] != 0:
            node_in_edge_num = node_in_edge_num + 1
    return node_in_edge_num


'''
超边强度
首先求
1、包含在两个边的点的个数
2、超边的邻接超边
'''


def hyperedge_strength(hg: hnx.Hypergraph, edge_i: int):
    matrix = hg.edge_adjacency_matrix().todense()  # 边的邻接矩阵：边边
    size = matrix.shape[0]
    node_num = 0
    for j in range(size):
        node_num = node_num + matrix[edge_i, j]
    return node_num
