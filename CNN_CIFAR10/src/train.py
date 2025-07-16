import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from CNN_CIFAR10.src.GetCIFAR10 import train_data
from CNN_CIFAR10.src.GetCIFAR10 import test_data
from CNN_CIFAR10.src.model import MyModel

#加载数据
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

#初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#一些参数
epochs = 10
train_loss = 0.0
train_acc = 0.0
test_loss = 0.0
test_acc = 0.0
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch+1}, loss: {train_loss:.4f}')

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {100 * test_acc:.2f}%')
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), '../models/CIFAR10.ckpt')
        print('saving model with test acc {:.3f}'.format(best_acc))