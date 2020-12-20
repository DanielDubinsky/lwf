import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from src.data.Cifar100_cifar10 import JointCifar100cifar10


def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False).cuda()
    # compute the log of softmax values
    outputs = torch.log_softmax(logits/T, dim=1)
    labels = torch.softmax(labels/T, dim=1)
    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)
    return Variable(outputs.data, requires_grad=True).cuda()


class MultiHeadLinear(nn.ModuleList):
    def __init__(self, in_features, out_features, bias=True, no_grad=True):
        super(MultiHeadLinear, self).__init__()

        self.out_features = out_features
        for _ in range(out_features):
            self.append(nn.Linear(in_features, 1, bias))

        # parameters for future tasks should not be trainable
        if no_grad:
            for param in self.parameters():
                param.requires_grad = False

    def set_trainable(self, ts=[]):
        if not isinstance(ts, (list, range)):
            ts = [ts]
        for t, m in enumerate(self):
            requires_grad = (t in ts)
            for param in m.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        y = torch.cat([self[t](x) for t in range(self.out_features)], dim=1)
        return y

# Kaiming initialization


def init_module(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)


def init_model(config):
    from .renset import ResNet
    args, kwargs = ResNet.get_args_from_config(config)
    net = ResNet(*args, **kwargs).cpu()

    return net


def load_model(path):
    loaded = torch.load(path, map_location=torch.device('cpu'))
    net = init_model(loaded['config'])
    net.load_state_dict(loaded['model'])
    print(loaded['config'])
    return net


def get_dataset(name: str, train: bool):
    if train:
        t = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                # mean=[0.5071, 0.4865, 0.4409] for cifar100
                mean=[0.4914, 0.4822, 0.4465],
                # std=[0.2009, 0.1984, 0.2023] for cifar100
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])
    else:
        t = torchvision.transforms.ToTensor()
    if name == 'cifar10':
        return torchvision.datasets.cifar.CIFAR10(
            root='./data/raw/cifar10', train=train, transform=t)
    if name == 'cifar100':
        return torchvision.datasets.cifar.CIFAR100(
            root='./data/raw/cifar100', train=train, transform=t)
    if name == 'cifar10&cifar100':
        if train:
            return JointCifar100cifar10(
                './data/raw/cifar10', './data/raw/cifar100', train=True, transform=t)
        else:
            ds10 = torchvision.datasets.cifar.CIFAR10(
                root='./data/raw/cifar10', train=False, transform=t)
            ds100 = torchvision.datasets.cifar.CIFAR100(
                root='./data/raw/cifar100', train=False, transform=t)
            return (ds10, ds100)
