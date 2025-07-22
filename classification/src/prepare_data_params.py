import gc

import torch
import numpy as np
from sympy.codegen.ast import int32

from classification import*
from LibriDataset import *
from model import *
# data prarameters
# 用于数据处理时的参数
config = {
    'concat_nframes': 3, # 要连接的帧数,n必须为奇数（总共2k+1=n帧）
    'train_ratio': 0.8,
    #training parameters
    'seed': 0, #随机种子
    'batch_size': 512,
    'num_epochs': 5,
    'learning_rate': 1e-3,
    'model_path':'./model.ckpt',
    #model parameters
    'hidden_layers':1,
    'hidden_dim':256
}

config['input_dim'] = 39 * config['concat_nframes']  # 模型的输入维度，不应更改该值，这个值由上面的拼接函数决定

#预处理数据
train_X, train_y = preprocess_data(split='train', feat_dir='../libriphone/feat', phone_path='../libriphone', concat_nframes=config['concat_nframes'], train_ratio=config['train_ratio'])
val_X, val_y = preprocess_data(split='val', feat_dir='../libriphone/feat', phone_path='../libriphone', concat_nframes=config['concat_nframes'], train_ratio=config['train_ratio'])

#导入数据
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

#删除原始数据以节省内存
del train_X, train_y, val_X, val_y
gc.collect() #清理不再使用的内存对象，释放内存

#用dataloader加载数据
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #得到的是torch.device对象
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#固定随机种子
def same_seed(seed):
    torch.manual_seed(seed) #为CPU设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #为当前GPU设置
        torch.cuda.manual_seed_all(seed) #为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

same_seed(config['seed'])

# 创建模型、定义损失函数和优化器
model = Classifier(input_dim=config['input_dim'], hidden_layers=config['hidden_layers'], hidden_dim=config['hidden_dim']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])


if __name__ == '__main__':

    #训练模型
    best_acc = 0.0
    for epoch in range(config['num_epochs']):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        #训练部分
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() #梯度清零
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, train_pred = torch.max(outputs, 1) # 获得概率最高的类的索引 返回两个张量，第一个是真正的最大值，第二个是对应的索引
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        #验证部分
        if len(val_set) > 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)
                    _, val_pred = torch.max(outputs, 1)
                    val_acc +=(val_pred.detach() == labels.detach()).sum().item()
                    val_loss += loss.item()
                    #输出整数，宽度3位，不足补0（如 001） 	输出浮点数，宽度3位，小数点后保留6位
                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, config['num_epochs'], train_acc / len(train_set), train_loss / len(train_loader),
                    val_acc / len(val_set), val_loss / len(val_loader)
                ))

                # 如果模型获得提升，在此阶段保存模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), config['model_path'])
                    print('saving model with val acc {:.3f}'.format(best_acc / len(val_set)))

        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, config['num_epochs'], train_acc / len(train_set), train_loss / len(train_loader)
            ))

        # 如果结束验证，则保存最后一个epoch得到的模型
        if len(val_set) == 0:
            torch.save(model.state_dict(), config['model_path'])
            print('saving model at last epoch')
            del train_loader, val_loader
            gc.collect()


    #测试
    test_X = preprocess_data(split='test', feat_dir='../libriphone/feat', phone_path='../libriphone', concat_nframes=config['concat_nframes'])
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    #加载已经训练好的模型
    model = Classifier(input_dim=config['input_dim'], hidden_layers=config['hidden_layers'], hidden_dim=config['hidden_dim']).to(device)
    model.load_state_dict(torch.load(config['model_path']))

    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            outputs = model(features)
            _, test_pred = torch.max(outputs, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    #将预测结果写入CSV文件
    with open('pred.txt', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{}, {}\n'.format(i, y))