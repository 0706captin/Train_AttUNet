import math
from numpy import *
import os
import pandas as pd
import numpy as np

def get_k_fold_data(k, i, X):  # 此过程主要是步骤（1）
    '''
    返回第i折交叉验证时所需要的训练和验证数据，
    分开放，X_train为训练数据，X_valid为验证数据
    X : images的名字，是个list， ['name1', 'name2', 'name3', ...]
    y : labels的名字，是个list， ['name1', 'name2', 'name3', ...]
    '''
    assert k > 1

    # 每份的个数:数据总条数/折数（组数）
    fold_size = len(X) // k  # ==100
    X_train = None, None
    X_valid = None, None

    for j in range(k):

        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = X[idx]

        if j == i:  # 第i折作valid
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = X_train + X_part
            # X_train.append(X_part)
    return X_train, X_valid


def get_rmse(x):
    """
    均方根误差：它的计算方法是先平方、再平均、然后开方
    """
    m = np.mean(x)
    sum = 0
    for j in range(len(x)):
        sum += (x[j] - m) ** 2
    sd = math.sqrt(sum / len(x))
    return sd


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

def save_score(k, name, mean_iou, mean_acc, dice, hd, date):
    data = {'name': name, 'mean_iou': mean_iou, 'mean_acc': mean_acc, 'dice': dice, 'hd': hd}
    dataframe = pd.DataFrame(data, columns=['name', 'mean_iou', 'mean_acc', 'dice', 'hd'])  # columns自定义列的索引值
    csv_file = os.path.join('./results/csv_file', date)
    mkfile(csv_file)
    csv_name = 'k' + str(k) + '_score' + '.csv'
    a = os.path.join(csv_file, csv_name)
    dataframe.to_csv(a)

if __name__ == '__main__':
    i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 5]
    rmse = get_rmse(i)
    print(rmse)
