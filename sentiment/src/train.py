import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentiment.src.build_vocab import Vocab
from sentiment.src.dataset import IMDBDataset
from sentiment.src.model import IMDBModel
import pickle


num_epochs = 10
learning_rate = 0.003
embedding_dim = 300
pad = Vocab.PAD
text_max_len = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = pickle.load(open('../models/vocab.pkl', 'rb'))
model = IMDBModel(num_embedding=len(vocab), embedding_dim=embedding_dim, pad=pad, text_max_len=text_max_len).to(device)

def collate_fn(batch):
    '''
    对batch数据进行预处理，将其转换为tensor
    '''
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([vocab.transform(i, max_len=text_max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels

def get_dataloader(train=True):
    dataset = IMDBDataset(train=train)
    if train:
        return DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

train_dataloader = get_dataloader(train=True)
test_dataloader = get_dataloader(train=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_bar = tqdm(train_dataloader)
    for idx , (review, label) in enumerate(train_bar): # idx 是 enumerate() 自动生成的索引，表示当前循环是第几次
        optimizer.zero_grad()
        review = review.to(device)
        label = label.to(device)
        outputs = model(review)
        loss = criterion(outputs, label)
        train_loss += loss.item() #item()就是将tensor转为float
        loss.backward()
        optimizer.step()
        train_bar.set_description('epoch:{} idx:{} train_loss:{:.6f}'.format(epoch+1, idx, train_loss)) #设置进度条前缀文本

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    best_loss = float('inf')
    test_bar = tqdm(test_dataloader)
    with torch.no_grad():
        for idx, (review, label) in enumerate(test_dataloader):
            review, label = review.to(device), label.to(device)
            outputs = model(review)
            loss = criterion(outputs, label)
            test_loss += loss.item()
            pred = torch.argmax(outputs, dim=1) #获取最大值的下标
            correct += (pred == label).sum().item()
            total += label.size(0)
        acc = 100 * correct / total
        print('epoch:{} accuracy:{:.6f}'.format(epoch, acc))
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), '../models/best_model.pt')
