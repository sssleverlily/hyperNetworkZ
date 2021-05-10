import numpy as np
import matplotlib.pyplot as plt
from Notebooks import micro_statistics
import hypernetx as hnx

'''
测试能不能画统计图
先从微观统计量开始
'''

'''
同一节点不同指标
'''


def small_draw(hg: hnx.Hypergraph, node: int):
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    statistic_value = []
    # 添加
    statistic_value.append(micro_statistics.hyperdegree(hg, node).real)
    statistic_value.append(micro_statistics.hyperdistribution(hg, node).real)
    statistic_value.append(micro_statistics.Subgraph_centrality(hg, node).real)
    statistic_value.append(micro_statistics.degree_centrality(hg, node).real)
    statistic_value.append(micro_statistics.hypernode_strength(hg, node).real)
    statistic_value.append(micro_statistics.cosine_degree(hg, node).real)
    statistic_value.append(micro_statistics.node_clustering_coefficient(hg, node).real)
    statistic_value.append(micro_statistics.node_type_entropy(hg).real)
    statistic_value.append(micro_statistics.type_assortativity_coefficient(hg).real)
    statistic_value.append(micro_statistics.point_intensity_centrality(hg, node).real)
    # statistic_value.append(smalltest.node_betweenness_centrality(hg, node))
    # statistic_value.append(smalltest.node_importance(hg, node, 1, 0, 0))
    labels = []
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    value_len = len(statistic_value)
    for i in range(value_len):
        labels.append(str(i))

    # men_means = [20, 34, 30, 35, 27]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # 条形的宽
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, statistic_value, width, label='values', edgecolor='grey',
                    linewidth='1')  # edgecolor条形图的边缘的颜色，linewidth边缘的宽度

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    autolabel(rects1, ax)

    fig.tight_layout()

    plt.show()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


'''
同一指标不同节点
'''


def draw_node_degree(hg: hnx.Hypergraph):
    matrix = hg.adjacency_matrix().todense()
    length = matrix.shape[0]
    statistic_value = []
    labels = []
    for i in range(length):
        # 添加
        # 在这里可以变换想要统计的量，但我还没想好怎么改成可变的。目前只能写死了
        statistic_value.append(micro_statistics.hyperdegree(hg, i).real)
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    value_len = len(statistic_value)
    for i in range(value_len):
        labels.append(str(i))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # 条形的宽
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, statistic_value, width, label='values', edgecolor='grey',
                    linewidth='1')  # edgecolor条形图的边缘的颜色，linewidth边缘的宽度
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 20)
    ax.legend()
    autolabel(rects1, ax)

    fig.tight_layout()
    plt.show()


'''
画网络结构图
'''


def hypergraphdraw(hg: hnx.Hypergraph):
    hnx.draw(hg)
    #在Notebooks文件夹下存一张filename的图片
    # plt.savefig("filename.png")
    plt.show()
    return
