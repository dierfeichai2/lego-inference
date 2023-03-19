from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader


def CIFAR10Dataloader(root_dir, batch_size=128, num_workers=8):
#数据集预处理与超参数设定
    #数据归一化：对数据按通道进行标准化，即先减均值，再除以标准差
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    #torchvision.transforms.Compose()类。这个类的主要作用是定义串联多个图片变换的操作。
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),           #随机裁剪32x32，上下左右均填充4piexl，即统一为40x40
        transforms.RandomHorizontalFlip(),              #依概率对图像进行水平翻转，默认概率为0，5
        transforms.ToTensor(),                          #将PIL Image转为tensor,并归一化至[0-1]（直接除以255）
        normalize,                                      #数据标准化
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    #调用CIFAR10方法，下载数据到传入的root_dir，
    train_set = CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)
    #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
