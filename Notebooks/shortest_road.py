"""
2021.3.23
试一下超图的最短路算法
主要其实还是要最短路径的数量
1、dijkstra标号法
2、暴力全部遍历一遍法
"""
from typing import List

import hypernetx as hnx
import math
from Notebooks import micro_statistics
from Notebooks import meso_statistics
import numpy as np
import scipy as sp

'''
初始化
'''


def init(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    # 定义一个连接矩阵
    connect_matrix = np.zeros(shape=(node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            if matrix[i, j] != 0:
                connect_matrix[i, j] = 1
    return connect_matrix


'''
开始计算
1、遍历 i 的邻接节点
'''


def shortest(hg: hnx.Hypergraph, node_i: int, node_j: int):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    connect_matrix = init(hg)
    # 距离矩阵，开始的时候没有在一条超边的都定义为无穷大，在的就是0
    distance_martrix = np.matrix(np.ones((node_num, node_num)) * np.inf)
    for i in range(node_num):
        for j in range(node_num):
            if connect_matrix[i, j] == 1:
                distance_martrix[i, j] = 1
    # 如果两个在同一条超边里面，直接返回0
    # if connect_matrix[node_i, node_j] == 1:
    #     return 0
    # # 如果不在，就要dijkstra计算
    # print(deepSearch(hg, node_i, node_j, 0))
    return dijkstra(distance_martrix, node_i, node_j)


'''
node的邻接节点们
'''


def adjacency_node(hg: hnx.Hypergraph, node: int):
    matrix = hg.adjacency_matrix().todense()
    node_num = matrix.shape[0]
    adjacency_nodes = []
    for i in range(node_num):
        if matrix[node, i] != 0:
            adjacency_nodes.append(i)
    return adjacency_nodes


'''
Dijkstra求解最短路径算法
输入：原始数据矩阵，起始顶点,终止顶点
输出；起始顶点到其他顶点的最短距离
'''


def dijkstra(data_matrix, start_node, end_node):
    vex_num = len(data_matrix)
    flag_list = ['False'] * vex_num
    prev = [0] * vex_num
    dist = ['0'] * vex_num
    for i in range(vex_num):
        flag_list[i] = False
        prev[i] = 0
        dist[i] = data_matrix[start_node, i]
    flag_list[start_node] = False
    dist[start_node] = 0

    k = 0
    for i in range(1, vex_num):
        min_value = 99999
        for j in range(vex_num):
            if flag_list[j] == False and dist[j] != 'N':
                min_value = dist[j]
                k = j
        flag_list[k] = True

        for j in range(vex_num):
            if data_matrix[k, j] == 'N':
                temp = 'N'
            else:
                temp = min_value + data_matrix[k, j]
            if flag_list[j] == False and temp != 'N' and temp < dist[j]:
                dist[j] = temp
                prev[j] = k
    result = dist[end_node]
    if dist[end_node] == math.inf:
        result = 0
    return result
