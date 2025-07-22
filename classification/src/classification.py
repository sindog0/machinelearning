import os

import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

#定义导入feature函数
def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x
    return torch.cat((left, right), dim=0)

# 将前后的特征联系在一起，如concat_n = 11 则前后都接5
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 #n必须为奇数
    if concat_n < 2:
        return x
    seq_len, feat_dim = x.size(0), x.size(1) #frame的个数和frame的维度
    x = x.repeat(1, concat_n) #沿着第1维，重复concat_n次
    '''
    view是重构张量的维度
    view:[seq_len, concat_n, feat_dim]
    permute是调换顺序
    permute:[concat_n, seq_len, feat_dim]
    '''
    x= x.view(seq_len, concat_n, feat_dim).permute(1, 0, 2)
    mid = ( concat_n // 2)
    for r_idx in range(1,mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feat_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41
    mode = 'train' if(split =='train' or split =='val') else 'test'
    label_dict = {} #用一个字典来存标签
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]
    if split == 'train' or split == 'val':
        #分割训练和验证数据集
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list] #去除每一行首尾的\n
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long) #创建一个长度为max_len，类型为int64的张量，并未进行初始化
    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])
        X[idx:idx + cur_len, :] = feat
        if mode != 'test':
            y[idx:idx + cur_len] = label
        idx += cur_len
    X =X[:idx, :]
    if mode != 'test':
        y = y[:idx]
    print(f'[INFO] {split} set')
    print(X.shape)

    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X
# 返回的X代表数据的维度，如果不链接则为39 如果链接即为n*39 n为连接的特征总数,y为标签

