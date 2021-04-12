import numpy as np
import pandas as pd

def data_wash():
    ecommerce_data = pd.read_csv('/Users/ssslever/PycharmProjects/hyperNetworkZ/Data/Ecommerce_data/test.csv')
    user_size = ecommerce_data['user_id'].shape[0]
    # for i in range(user_size):

    return




if __name__ == '__main__':
    data_wash()