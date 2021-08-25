import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import pandas as pd
import copy

def DataProcess():
    '''
    :function:  从csv文件当中读取数据
                划分为训练即和验证集
                处理成可迭代tuple类型数据集
    :return: train_data 训练使用的数据集
            ,test_data  测试/验证使用数据集
    '''
    x = np.loadtxt('./dataset/data.csv')
    y= np.loadtxt('./dataset/label.csv')
    x=x.reshape(28709,1,48,48)
    X_train=x[0:20001]
    X_test=x[20001:]
    y_train=y[0:20001]
    y_test=y[20001:]

    X_train_tor=torch.from_numpy(X_train.astype(np.float32))
    X_test_tor=torch.from_numpy(X_test.astype(np.float32))
    y_train_tor=torch.from_numpy(y_train.astype(np.int64))
    y_test_tor=torch.from_numpy(y_test.astype(np.int64))

    train_data=Data.TensorDataset(X_train_tor,y_train_tor)
    test_data=Data.TensorDataset(X_test_tor,y_test_tor)

    return train_data,test_data


def split_dataset(dataset, rank ,seed=3):
    length=len(dataset) // 4
    print('lenth :' ,length,'rank : ',rank)
    lengths=[length]*4
    delete_num = len(dataset) % 4
    if delete_num != 0:
        lengths.append(delete_num)
    return torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))[rank]

def make_dataloader(train_dataset,test_dataset):
    train_loader=Data.DataLoader(
        dataset=train_dataset
        ,batch_size=64
        , shuffle=True
        , num_workers = 4
        , pin_memory = True
    )
    test_loader=Data.DataLoader(
        dataset=test_dataset
        ,batch_size=64
        , shuffle=True
        , num_workers=4
        , pin_memory=True
    )
    return train_loader,test_loader