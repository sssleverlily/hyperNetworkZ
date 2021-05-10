# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
import hypernetx as hnx
from Notebooks import micro_statistics
from Notebooks import meso_statistics
from Notebooks import macro_statistics
from Notebooks import hyperdraw

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
    print(edges)
    HG = hnx.Hypergraph(dict(enumerate(edges)))
    hyperdraw.hypergraphdraw(HG)
    hyperdraw.draw_node_degree(HG)
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
    #随机生成超图
    micro_statistics.hyperdistribution(hg, 3)
    edges = hg.edges
    nodes = hg.nodes
    #节点交换矩阵
    node_matrix = np.zeros((len(nodes), len(nodes)))
    # print(edges)
    for i in hg.nodes:
        for j in hg.nodes:
            print(j)
            if micro_statistics.hyperdistribution(hg, micro_statistics.hyperdegree(hg, int(i))) \
                    == micro_statistics.hyperdistribution(hg,micro_statistics.hyperdegree(hg, int(j))):
                node_matrix[int(i)][int(j)] = 1 #代表可以交换
    # print(edges)
    # hnx.Hypergraph(dict(enumerate()))

    return
