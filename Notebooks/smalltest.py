# 超度，节点的超度分布，节点的子图中心度，度中心性，点强度
import hypernetx as hnx
import networkx as nx
from scipy.sparse import csr_matrix
import scipy as sp
import scipy.sparse.linalg
import math


# 邻接矩阵
def adjacency_matrix(hg: hnx.Hypergraph):
    print(hg.adjacency_matrix())


'''
输入:hg:超图，node:节点
输出：degree：节点的超度
'''


def hyperdegree(hg: hnx.Hypergraph, node: int):
    node_list = list(hg.nodes.elements.keys())  # 获得超图节点列表
    node_degree = hg.degree(node_list[node])
    # print(node_degree)
    return node_degree


'''
输入:hg:超图，node_degree_is_k:度为几
输出：distribution：超度分布
    node_list：所有节点的值的集合
    遍历，如果该节点的度为node_degree_is_k，则录入统计
'''


def hyperdistribution(hg: hnx.Hypergraph, node_degree_is_k: int):
    nodes_Sum = 0
    node_list = list(hg.nodes.elements.keys())  # 获取超图的节点列表
    nodes_Num = len(hg.nodes)
    for i in range(nodes_Num):
        if hg.degree(node_list[i]) == node_degree_is_k:
            nodes_Sum = nodes_Sum + 1
    distribution = nodes_Sum / nodes_Num
    print(distribution)


'''
输入:hg:超图，node:点
输出：sub_centrality：子图中心度
'''


def Subgraph_centrality(hg: hnx.Hypergraph, node: int):
    # a, b = np.linalg.eig(keykey)
    martix = hg.adjacency_matrix().asfptype()  # 上投矩阵以浮动或翻倍
    vals, vecs = sp.sparse.linalg.eigs(martix)  # val特征值，vecs特征向量
    sub_centrality = 0
    for i in range(6):
        sub_centrality = sub_centrality + vecs[node][i] * vecs[node][i] * math.exp(vals[i])
    print(sub_centrality)
    return sub_centrality


'''
输入:hg:超图，node:点
输出：degree_centrality：度中心性
遍历邻接矩阵，如果有直接相连的节点，则录入计算direct_node_sum
'''


def degree_centrality(hg: hnx.Hypergraph, node: int):
    matrix = hg.adjacency_matrix().todense()
    nodes_Num = len(hg.nodes)
    size = matrix[node].size
    direct_node_sum = 0
    for i in range(size):
        if matrix[node, i] != 0:
            direct_node_sum = direct_node_sum + 1
    degree_centrality = direct_node_sum / (nodes_Num - 1)
    print(degree_centrality)


'''
输入:hg:超图，node:点
输出：node_strength：节点强度
遍历关联矩阵的行（点），再遍历关联矩阵的列（边），如果node节点与正在遍历的点同时在一条边中，则录入node_strength统计
'''


# 将节点i与节点j的权值定义为同时包含节点i与节点j的超边数量
def hypernode_strength(hg: hnx.Hypergraph, node: int):
    matrix = hg.incidence_matrix().todense()  # 关联矩阵
    node_num = matrix.shape[0]
    edge_num = matrix.shape[1]
    node_strength = 0
    for j in range(node_num):
        for i in range(edge_num):
            if matrix[node, i] == 1 and matrix[j, i] == 1:
                node_strength = node_strength + 1
    print(node_strength)


'''
输入:hg:超图，node:点
输出：cos_degree：节点的余平均度
节点i的余平均度定义为节点i的所有邻居节点的平均超度
遍历节点i的邻接节点，并统计其度的平均值即可
'''


def cosine_degree(hg: hnx.Hypergraph, node: int):
    matrix = hg.adjacency_matrix().todense()
    size = matrix[node].size
    degree_num = 0
    direct_node_sum = 0
    for i in range(size):
        if matrix[node, i] != 0:
            degree_num = degree_num + hg.degree(str(node))
            direct_node_sum = direct_node_sum + 1
    cos_degree = degree_num / direct_node_sum
    print(cos_degree)


'''
输入:hg:超图，node:点
输出：node_clustering：节点的聚类系数
遍历邻接矩阵的第i行，找出i的邻接节点，再找出共有的超边数量
除以最大可能的边数即可
'''


def node_clustering_coefficient(hg: hnx.Hypergraph, node: int):
    adjacency_matrix = hg.adjacency_matrix().todense()  # 邻接矩阵
    node_num = adjacency_matrix.shape[0]
    node_edge_shared = 0
    node_direct_num = 0
    # 先找邻接节点
    for i in range(node_num):
        if (adjacency_matrix[node, i]) != 0:
            node_edge_shared = node_edge_shared + adjacency_matrix[node, i]
            node_direct_num = node_direct_num + 1
    node_direct_num = (node_direct_num * (node_direct_num)) / 2
    node_clustering = node_edge_shared / node_direct_num
    print(node_clustering)
    return node_clustering


'''
节点类型熵
'''


def node_type_entropy(hg: hnx.Hypergraph):
    node_list = list(hg.nodes.elements.keys())  # 获得超图节点列表
    list_size = len(node_list)
    n_i = 0
    nte = 0
    for node_type in range(list_size):
        for i in range(list_size):
            if hg.degree(node_list[i]) == node_type:  # 这里有问题，应该是节点的type而不是degree，但是不知道怎么表示type
                n_i = n_i + 1
        p_i = n_i / list_size
        if p_i != 0:
            nte = nte + -1 * p_i * math.log(p_i, math.e)
    print(nte)


'''
节点类型匹配系数（TAC）
'''


def type_assortativity_coefficient(hg: hnx.Hypergraph):
    node_list = list(hg.nodes.elements.keys())  # 获得超图节点列表
    list_size = len(node_list)
    n_p = 0
    n_pp = 0
    # npp
    for p in range(list_size):
        for i in range(list_size):
            if hg.degree(node_list[i]) == p:  # 这里有问题，应该是节点的type而不是degree，但是不知道怎么表示type
                n_p = n_p + 1
        n_pp = n_pp + n_p * (n_p - 1)

    # ap和bp
    an_q = 0
    an_p = 0
    a_p = 0
    a_pp = 0
    for p in range(list_size):
        for q in range(list_size):
            for i in range(list_size):
                if hg.degree(node_list[i]) == q:  # 这里有问题，应该是节点的type而不是degree，但是不知道怎么表示type
                    an_q = an_q + 1
                if hg.degree(node_list[i]) == p:  # 这里有问题，应该是节点的type而不是degree，但是不知道怎么表示type
                    an_p = an_p + 1
            if p == q:
                a_p = a_p + an_q * (an_q - 1)
            elif p != q:
                a_p = a_p + an_q * an_p
    a_pp = a_pp + a_p * a_p

    tac = (n_pp - a_pp) / (1 - a_pp)
    print(tac)


'''
点强度中心性
两个节点的连接强度定义为同时包含这两个节点的超边数量
一个节点所有邻接节点的连接强度之和
'''


def point_intensity_centrality(hg: hnx.Hypergraph, node: int):
    node_matrix = hg.adjacency_matrix().todense()
    node_size = node_matrix.shape[0]
    node_strength = 0
    j = 0
    # #先把邻接节点找出来
    for i in range(node_size):
        # if node_matrix[node, i] != 0: 其实判不判断是邻接节点没必要，反正是+0
        node_strength = node_strength + node_matrix[node, i]
    print(node_strength)


'''
节点的介数中心性
'''


def node_betweenness_centrality(hg: hnx.Hypergraph, node: int, s=1, normalized=True):
    A, coldict = hg.adjacency_matrix(index=True)
    # A, coldict = hg.edge_adjacency_matrix(s=s, index=True)
    A.setdiag(0)
    A = (A >= s) * 1
    g = nx.from_scipy_sparse_matrix(A)
    dict = nx.betweenness_centrality(g, normalized=normalized)
    # print(dict.get(node - 1))
    return dict.get(node - 1)


'''
节点重要性
a,b,c 是调节参数，只要满足a+b+c=1即可
'''


def node_importance(hg: hnx.Hypergraph, node: int, a: int, b: int, c: int):
    if a+b+c != 1:
        print("输入参数错误")
    else:
        node_matrix = hg.adjacency_matrix().todense()
        node_num = node_matrix.shape[0]
        edge_matrix = hg.incidence_matrix().todense()
        edge_num = edge_matrix.shape[0]
        print(edge_matrix.shape)
        node_degree = 0
        node_star = 0
        for i in range(node_num):
            if node_matrix[node, i] != 0:
                node_degree = node_degree + 1
        for i in range(edge_num):
            if edge_matrix[i, node] != 0:
                node_star = node_star + 1
        node_betweenness = node_betweenness_centrality(hg, node)
        node_importance = a * node_degree + b * node_star + c * node_betweenness
        print(node_importance)
    return
