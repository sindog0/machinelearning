import os
import torch
from torch.utils.data import DataLoader

# 从各自的文件中导入模块
from model import Classifier
from data import FoodDataset, train_tfm, test_tfm
from  function import trainer

# 设备配置 (优先使用 GPU，否则使用 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用设备: {device}")

# 配置字典
config = {
    'batch_size': 64,
    'num_epochs': 5,
    'early_stop_count': 300, # 注意：对于 5 个 epoch 来说，这个值非常高，可能永远不会触发早停
    'seed': 6666,
    'dataset_dir': '../food11', # 假设数据集在当前脚本的父目录
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'save_path': '../models/model.ckpt',
    'resnet_save_path': '../models/resnet_model.ckpt',
    'resnet_flag': False # 标志，指示是否使用 ResNet 模型 (当前代码使用自定义 Classifier)
}

# --- 数据加载 ---
_dataset_dir = config['dataset_dir']

# 训练集
train_set = FoodDataset(os.path.join(_dataset_dir, 'training'), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)

# 验证集 (注意：你目前对验证集也使用了 train_tfm，这意味着包含数据增强。
# 通常，验证集和测试集只进行缩放和ToTensor，不进行数据增强，以获得更真实的性能评估。)
valid_set = FoodDataset(os.path.join(_dataset_dir, 'validation'), tfm=train_tfm)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True)

# 测试集 (测试集通常不打乱顺序，以便结果可复现)
test_set = FoodDataset(os.path.join(_dataset_dir, 'test'), tfm=test_tfm) # 测试集使用 test_tfm
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

# --- 模型初始化和训练 ---
model = Classifier().to(device)

# 确保模型保存目录存在
os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
os.makedirs(os.path.dirname(config['resnet_save_path']), exist_ok=True) # 即使不使用 ResNet 也创建，以防将来启用

# 开始训练
trainer(train_loader, valid_loader, model, config, device, resnet_flag=config['resnet_flag'])