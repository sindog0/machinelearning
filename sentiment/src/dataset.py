import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import re #正则



def tokenize(text):
    """
    进行文本分词
    :return: [str, str, str]
    """
    # filters = ['!', '"', '@', '#', '$', '%', '^', '&', '*', '\(', '\)', '\*', '\+', '\.', '/', ':', ';',
    #            '<', '>', '\?', '\\', '\]', '_', '~', '\t', '\n', '\x97', '\x96']
    filters = ['!', '"', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '.', '/', ':', ';',
               '<', '>', '?', '\\', ']', '_', '~', '\t', '\n', '\x97', '\x96']
    text = text.lower() #将大写转换为小写
    text = re.sub('<br />', ' ', text)
    #re.escape() 会自动把所有特殊字符转义，比如 * 变成 \*，? 变成 \?，防止正则错误。
    text = re.sub('|'.join(re.escape(f) for f in filters), ' ', text) #按照filters的规则替换为空格
    ret = [i for i in text.split(' ') if len(i) > 0] #然后再按空格进行分割

    return ret

def collate_fn(batch):
    #使每一个句子的size相同
    reviews, labels = zip(*batch)
    return reviews, labels

class IMDBDataset(Dataset):
    def __init__(self, train=True):
        super(IMDBDataset, self).__init__()
        data_path = '../data/IMDB'
        if train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'test')
        self.total_path = [] #用列表保存所有文件路径
        for temp_path in ['neg', 'pos']:
            cur_path = os.path.join(data_path, temp_path)
            self.total_path +=[os.path.join(cur_path, i) for i in os.listdir(cur_path)]

    def __getitem__(self, idx):
        file = self.total_path[idx]
        #从txt获取评论并分词
        review = tokenize(open(file, 'r').read())
        #获取评论对应的label
        label =int(file.split('_')[-1].split('.')[0]) #文件名类似 0_3.txt  ['0', '3.txt']
        label = 0 if label < 5 else 1 #低于5分为neg
        return review, label

    def __len__(self):
        return len(self.total_path)

if __name__ == '__main__':
    imdb = IMDBDataset()
    dataloader = DataLoader(imdb, batch_size=64, shuffle=True, collate_fn=collate_fn)
    for data in dataloader:
        print(data)
