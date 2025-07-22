import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
#绘制pytorch的网络
from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter



device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # 随机种子，可以自己填写. :)
    'select_all': True,   # 是否选择全部的特征
    'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'epochs': 3000,     # 数据遍历训练次数
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # 如果early_stop轮损失没有下降就停止训练.
    'save_path': './models/model.ckpt'  # 模型存储的位置
}


def same_seed(seed):
    '''
    设置随机数（便于复现）
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed) #在CPU上设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # print(f'Set Seed = {seed}')
    print("Set Seed = {}".format(seed))


def train_valid_split(data_set, valid_ratio, seed):
    '''
    数据集拆分成训练集和验证集
    '''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

#数据集
class COVID19Dataset(Dataset):
    '''
    x: np.ndarray  特征矩阵.
    y: np.ndarray  目标标签, 如果为None,则是预测的数据集
    '''
    def __init__(self, x, y = None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


#神经网络模型
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    特征选择
    选择较好的特征用来拟合回归模型
    '''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]# 取所有行的最后一列（标签列）
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data #取所有行的前 n-1 列（特征列）

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: 选择需要的特征 ，这部分可以自己调研一些特征选择的方法并完善.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def trainer(train_loader, valid_loader, model, config, device):
    loss_fn = nn.MSELoss(reduction='mean')
    #优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    writer = SummaryWriter("logs")
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    epochs, best_loss, step, early_stop_count = config['epochs'], math.inf, 0, 0

    for epoch in range(epochs):
        model.train()
        loss_record = []

        # tqdm可以帮助我们显示训练的进度 本质上还是loader只是加了点东西
        train_pbar = tqdm(train_loader, position=0, leave=True)
        #设置进度条的左边:显示第几个epoch
        train_pbar.set_description(f'Epoch [{epoch+1}/{epochs}]')

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y =x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())

            # 训练完一个batch的数据，将loss 显示在进度条的右边
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch[{epoch+1}/{epochs}]:Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar("Loss/valid", mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) #模型保存
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

#导入数据集
same_seed = config['seed']

# 训练集大小(train_data size) : 2699 x 118 (id + 37 states + 16 features x 5 days)
# 测试集大小(test_data size）: 1078 x 117 (没有label (last day's positive rate))
pd.set_option('display.max_column', 200) #设置显示数据的列数
train_df, test_df = pd.read_csv('./data/covid.train.csv'), pd.read_csv('./data/covid.test.csv')
# display(train_df.head(5))
train_data, test_data = train_df.values, test_df.values
del train_df, test_df # 删除数据减少内存占用
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# 打印数据的大小
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# 特征选择
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# 打印出特征数量.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# 使用Pytorch中Dataloader类按照Batch将数据集加载
#pin_memory 是 torch.utils.data.DataLoader 的一个参数，用于提高 GPU 训练时 数据加载到显存的速度
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#开始训练
model = MyModel(input_dim=x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config, device)


#测试
preds = predict(test_loader, model, device)
test_preds = pd.DataFrame({'id': np.arange(len(preds)), 'tested_positive': preds})
test_preds.to_csv('./test_preds.csv', index=False)
