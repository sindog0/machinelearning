import torch
import torch.nn as nn
import torch.nn.functional as F


class IMDBModel(nn.Module):
    def __init__(self, num_embedding, embedding_dim, pad, text_max_len):
        super(IMDBModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=pad)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(in_features=2 * embedding_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        '''
        x: (batch_size, max_len)
        '''
        x_embeded = self.embedding(x) #[batch_size, max_len, embedding_dim]

        #全连接层需要二维的矩阵
        # x.size(0) == x.shape[0] 等价
        #view() 是 PyTorch 中用于改变张量形状的方法，相当于 NumPy 的 reshape()
        # x_embeded_viewed = x_embeded.view(x_embeded.size(0), -1)

        output, (hn, cn) = self.lstm(x_embeded)  #hn/cn 的 batch 维始终在中间 [num_layers * num_directions, batch_size, hidden_size]
        # print(output.shape) #[64, 200, 600]
        # print(hn.shape) #[4, 64, 300]
        # print(cn.shape) #[4, 64, 300]

        out = torch.cat([hn[-1, :, :], hn[-2, :, :]], dim=-1) #拼接正向最后一个输出和反向最后一个输出, dim=-1 表示在最后一个维度拼接,-2 表示倒数第二个元素
        #全连接
        out_fc1 = self.fc1(out)
        out_fc1_relu = F.relu(out_fc1)

        out_fc2 = self.fc2(out_fc1_relu)
        out_fc2_relu = F.relu(out_fc2)
        return  out

