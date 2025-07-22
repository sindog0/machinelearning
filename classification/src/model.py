import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module): #模块化思维
    def __init__(self, input_dim, output_dim):
        super(BasicBlock,self).__init__()
        self.block = nn.Sequential( #可以视作封装了一层hidden_layer
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=3, hidden_dim=256): #这里的hidden_layers就是要往神经网络中添加多少个BasicBlock
        super(Classifier,self).__init__()
        '''
        展开写
        temp = []
        for _ in range(hidden_layers)
            temp.append(BasicBlock(input_dim, hidden_dim))
        '''
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim), #输入层
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)], # *[]将循环得到的 解压 隐含层
            nn.Linear(hidden_dim, output_dim) #输出层
        )

    def forward(self, x):
        x = self.fc(x)
        return x
