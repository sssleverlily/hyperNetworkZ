# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
import hypernetx as hnx
from Notebooks import micro_statistics
from Notebooks import meso_statistics
from Notebooks import macro_statistics
from Notebooks import hyperdraw
import math
import networkx as nx

'''
er图的意思就是随机连接
比如：
边1 连接3个点
边2 连接5个点
边3 连接2个点
先实现这个随机图
[{'2', '0'}]
'''


def init_erNet():
    # 首先随机三个边
    edges = []
    nodes_num = [3, 5, 2, 1, 4, 6, 6, 7, 4, 2, 3, 2, 4, 1, 2, 3, 4, 6, 7, 7, 8]
    i = 0
    # str(random.randint(1, 7))
    # 先创建边
    for i in range(nodes_num.__len__()):
        edge = []
        for j in range(nodes_num[i]):
            edge.append(str(random.randint(1, 7)))
        edges.append(edge)
    # print(edges)
    HG = hnx.Hypergraph(dict(enumerate(edges)))
    # hyperdraw.hypergraphdraw(HG)
    # hyperdraw.draw_node_degree(HG)
    return HG
    # print(micro_statistics.degree_centrality(HG, 3))


# class ER_network:
#     def __init__(self, N, p, title):
#         self.num = N  # 初始时网络有 N 个节点
#         self.p = p  # 每对节点以概率 p 被选择，进行连边，不允许重复连边。
#         self.title = title
#
#     def Create_ER_network(self):
#         # 初始化矩阵
#         ER_matrix = np.zeros([self.num, self.num])
#         matrix_num = np.arange(self.num)
#
#         for i in matrix_num:
#             # 只对上三角矩阵进行判断概率是否应该连线
#             del_list = np.arange(i + 1)
#             matrix_num_del_i = np.delete(matrix_num, del_list)
#             for j in matrix_num_del_i:
#                 if (self.p >= random.random()):
#                     ER_matrix[i][j] = 1
#
#         # 翻转上三角矩阵至下三角，形成对称矩阵
#         ER_matrix += ER_matrix.T - np.diag(ER_matrix.diagonal())
#         return ER_matrix
#
#     # print(ER_matrix)
#
#     def element_sum(self, mat):
#         # 统计数组中所有元素出现的次数 ,返回字典形式
#         y = mat.sum(axis=1)
#         y = np.array(y)
#         key = np.unique(y)
#         result = {}
#         for k in key:
#             mask = (y == k)
#             y_new = y[mask]
#             v = y_new.size
#             result[k] = v
#         return result
#
#     def plot_degree_map(self, ele_sum, title):
#         mat_degree_percent = [key for key, value in ele_sum.items()]
#         mat_degree_percent1 = [value for key, value in ele_sum.items()]
#         mat_degree_percent2 = \
#             np.array(mat_degree_percent1) / sum(mat_degree_percent1)
#
#         x = mat_degree_percent
#         y = mat_degree_percent2
#
#         plt.plot(x, y, marker='o', mec='r', mfc='w', label='Degree map')
#         plt.legend()
#         plt.xlabel("degree")  # X轴标签
#         plt.ylabel("P(degree)")  # Y轴标签
#         plt.title(title)  # 标题
#         plt.show()
#
#     def main(self):
#         ER_matrix = self.Create_ER_network()
#         ele_sum = self.element_sum(ER_matrix)
#         self.plot_degree_map(ele_sum, self.title)
#
#
#
'''
一阶零模型
保证超度分布不变的零模型
'''


def one_order(hg: hnx.Hypergraph):
    # 随机生成超图
    micro_statistics.hyperdistribution(hg, 3)
    edges = hg.edges.elements
    nodes = hg._nodes._elements
    # node_num = max(nodes.values())
    # 节点交换矩阵
    node_matrix = np.zeros((len(nodes), len(nodes)))
    print(edges)
    for i in hg.nodes:
        for j in hg.nodes:
            if micro_statistics.hyperdistribution(hg, hg.degree(i)) \
                    == micro_statistics.hyperdistribution(hg, hg.degree(j)):
                print(hg._edges._elements[0]._elements)
                # 代表可以交换，应该是边换
    print(edges)
    # hnx.Hypergraph(dict(enumerate()))
    return


'''
计算srp
'''


def cal_srp(hx: hnx.Hypergraph):
    #随机生成九个零模型
    null_models = []
    summ = [0.0]*6
    for i in range(6):
        null_models.append(init_erNet())
    Nrandi = [0.0]*6
    Nreal = [0.0]*6
    delta_i = [0.0]*6
    sum_delta = 0
    srp_i = [0.0]*6
    #计算部分
    for i in range(6):
        summ[0] = summ[0] + macro_statistics.net_clustering_coefficient(null_models[i])
        # summ[1] = summ[1] + macro_statistics.net_subgraph_centrality(null_models[i])
        summ[1] = summ[1] + macro_statistics.average_shortest_path(null_models[i])
        summ[2] = summ[2] + macro_statistics.shannon_entropy(null_models[i])
        summ[3] = summ[3] + macro_statistics.hyperNet_efficiency(null_models[i])
        # summ[5] = summ[5] + macro_statistics.hyperNet_natural_connectivity(null_models[i])
        summ[4] = summ[4] + macro_statistics.classification_entropy(null_models[i])
        summ[5] = summ[5] + macro_statistics.hypernet_motif_entropy(null_models[i])
        # summ[8] = summ[8] + macro_statistics.hyperNet_centrality(null_models[i])

    Nreal[0] = macro_statistics.net_clustering_coefficient(hx)
    # Nreal[1] = macro_statistics.net_subgraph_centrality(hx)
    Nreal[1] = macro_statistics.average_shortest_path(hx)
    Nreal[2] = macro_statistics.shannon_entropy(hx)
    Nreal[3] = macro_statistics.hyperNet_efficiency(hx)
    # Nreal[5] = macro_statistics.hyperNet_natural_connectivity(hx)
    Nreal[4] = macro_statistics.classification_entropy(hx)
    Nreal[5] = macro_statistics.hypernet_motif_entropy(hx)
    # Nreal[8] = macro_statistics.hyperNet_centrality(hx)

    for i in range(6):
        Nrandi[i] = summ[i] / 6
        delta_i[i] = (Nreal[i]-Nrandi[i])/(Nreal[i]+Nrandi[i])
    for i in range(6):
        sum_delta = sum_delta + delta_i[i]*delta_i[i]
    for i in range(6):
        srp_i[i] = delta_i[i]/math.sqrt(sum_delta)
        print(srp_i[i])
    return


if __name__ == '__main__':
    net = nx.read_gml('/Users/ssslever/PycharmProjects/hyperNetworkZ/Data/netscience.gml')
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
    # print(len(edges_num))
    # print(len(nodes))
    HG = hnx.Hypergraph(dict(enumerate(smallEdges)))
    cal_srp(HG)