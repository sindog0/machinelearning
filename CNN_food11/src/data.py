import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from function import quick_observe # 从 src 导入 quick_observe

# 数据准备 (数据转换，数据扩增)
# 假设 food11/training 目录位于 data.py 的父目录
train_dir_root = '../food11/training'
quick_observe(train_dir_root) # 调用辅助函数观察目录

# 测试集转换：只进行缩放和转换为 Tensor
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 训练集转换：缩放、数据增强并转换为 Tensor
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # 数据增强
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
])


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # 测试集没有label
        return im, label