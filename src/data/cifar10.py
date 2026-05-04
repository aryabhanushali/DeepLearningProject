import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_cifar10_loaders(data_dir, batch_size, num_workers=0, augment=True, subset_frac=1.0):
    normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if subset_frac < 1.0:
        n = int(len(train_set) * subset_frac)
        indices = torch.randperm(len(train_set))[:n]
        train_set = Subset(train_set, indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
