import torch
from torch import nn
from torchvision.models import resnet50 # 导入 ResNet50
from function import model_plot # 从 src 导入 model_plot

# ResNet50 实例 (当前未预训练)
resnet = resnet50(pretrained=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 输入: [3, 128, 128] (通道数, 高, 宽)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 输出: [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出: [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), # 将 CNN 输出展平后连接到全连接层
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)  # 假设 food11 数据集有 11 个类别
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)  # -1 自动计算剩余维度，使形状变成 [batch_size, 特征数]
        x = self.fc(x)
        return x

# 模型的简单测试和绘图 (如果你想运行，可以取消注释)
input_dummy = torch.randn(1, 3, 128, 128).requires_grad_(True)
model_plot(Classifier, input_dummy)