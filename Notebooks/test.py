import hypernetx as hnx
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.collections import RegularPolyCollection


def init():
    """
    134 126 246 123 346 345 456 256
    :return:
    """
    edges = [['1', '3', '4'], ['1', '2', '6'], ['2', '4', '6'], ['1', '2', '3'], ['3', '4', '6'], ['3', '4', '5'],
             ['4', '5', '6'], ['2', '5', '6']]
    pos = {'3': (0, 2), '4': (1, 2 - math.sqrt(3)), '5': (3, 2 - math.sqrt(3)), '6': (4, 2), '2': (3, 2 + math.sqrt(3)),
           '1': (1, 2 + math.sqrt(3))}

    # collection = RegularPolyCollection(
    #     numsides=5,  # a pentagon
    #     rotation=0, sizes=(50,),
    #     edgecolors=(black,),
    #     linewidths=(1,),
    #     offsets=offsets,
    # )
    # np.arange(len(HG.edges)) % 8
    #设定颜色的代码  plt.cm.tab10(np.arange(len(H.edges))%10)
    black = (0, 0, 0, 1)
    # edges_kwargs.setdefault('edgecolors', plt.cm.tab10(np.arange(len(H.edges))%10))
    HG = hnx.Hypergraph(dict(enumerate(edges)))
    collection = {
        "linewidths": 3,
        "edgecolors": {
            (0, 0, 0, 1),
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0.6, 0.12, 0.95, 1),
            (0, 0, 1, 1),
            (0.12, 0.56, 1, 1.0),
            (0.93, 0.46, 0, 1.0),
            (0.32, 1, 0.62, 1),
            (0, 0.54, 0, 1.0)
        },
    }
    matplotlib.rcParams.update({'font.size': 25})
    hnx.draw(HG, pos=pos, edges_kwargs=collection)
    plt.show()
