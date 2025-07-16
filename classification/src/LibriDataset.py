import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#定义数据集，重写__init__ __getitem__ __len__方法
class LibriDataset(Dataset):#继承Dataset
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
