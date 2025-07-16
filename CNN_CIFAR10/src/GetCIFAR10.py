from torchvision import transforms, datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

train_data = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)

print(train_data.data.shape)