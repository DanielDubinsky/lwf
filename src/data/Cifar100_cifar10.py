import torch
import torchvision

class JointCifar100cifar10(torch.utils.data.Dataset):
    def __init__(self, cifar10_root: str, cifar100_root: str, train: bool, transform=None):
        self.cifar10 = torchvision.datasets.cifar.CIFAR10(root=cifar10_root, train=train, transform=transform)
        self.cifar100 = torchvision.datasets.cifar.CIFAR100(root=cifar100_root, train=train, transform=transform)

    def __len__(self):
        return len(self.cifar10) + len(self.cifar100)

    def __getitem__(self, idx):
        if idx < len(self.cifar10):
            return self.cifar10[idx]
        else:
            idx = idx - len(self.cifar10)
            image, label = self.cifar100[idx]
            return image, label + 10