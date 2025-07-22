import os.path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentiment.src.dataset import IMDBDataset
import pickle
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    reviews, labels = zip(*batch)
    return reviews, labels
def get_dataloader(train=True):
    dataset = IMDBDataset(train=train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    return dataloader
class Vocab:
    UNK_TAG = '<UNK>' #表示未知字符
    PAD_TAG = '<PAD>' #表示填充符
    UNK = 1
    PAD = 0

    def __init__(self):
        self.dict = { #保存词语和对应的数字
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }

        self.count =  {} #统计词频

    def fit(self, text):
        '''
        接收文本，统计词频（分词之后的文本）
        '''
        for token in text:
            self.count[token] = self.count.get(token, 0) + 1

    def build_vocab(self, min_len=1, max_len=None, max_feature=None):
        '''
        根据条件构建词表
        :param min_len:最小词频
        :param max_len:最大词频
        :param max_feature:最大词语数
        '''
        if min_len is not None:
            self.count = {token: count for token, count in self.count.items() if count >= min_len}
        if max_len is not None:
            self.count = {token: count for token, count in self.count.items() if count <= max_len}
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])

        for token in self.count:
            self.dict[token] = len(self.dict) #每次token对应一个数字，每加进来一个dict的长度都会加1
        #把dict进行翻转
        #zip将多个可迭代对象按位置一一配对，打包成元组
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, text, max_len=None):
        '''
        把文本转换成数字序列
        '''
        if len(text) > max_len:
            text = text[:max_len] #隔断
        else:
            text = text + [self.PAD_TAG] * (max_len - len(text)) #填充PAD

        return [self.dict.get(i, 1) for i in text]

    def inverse_transform(self, indices):
        '''
        把数字转换成文本
        '''
        return [self.inverse_dict.get(i, '<UNK>') for i in indices]

    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    vocab = Vocab() #实例化
    dl_train = get_dataloader(train=True)
    dl_test = get_dataloader(train=False)
    for reviews, labels in tqdm(dl_train):
        for text in reviews:
            vocab.fit(text)
    for reviews, labels in tqdm(dl_test):
        for text in reviews:
            vocab.fit(text)
    vocab.build_vocab()
    print(len(vocab))
    if not os.path.exists('../models'):
        os.mkdir('../models')
    pickle.dump(vocab, open('../models/vocab.pkl', 'wb'))