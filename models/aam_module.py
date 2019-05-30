import torch
import torch.nn as nn
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc1   = nn.Linear(in_planes, in_planes // 16)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.fc2   = nn.Linear(in_planes // 16, in_planes)
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc2(self.relu1(self.fc1(0.5 * self.avg_pool(x) + 0.5 * self.max_pool(x))))
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d()
        self.max_pool = nn.AdaptiveMaxPool2d()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = 0.5 * self.avg_pool(x) + 0.5 * self.max_pool(x)
        out = self.conv1(out)
        out = self.conv1(out)
        out = self.conv1(out)
        return self.sigmoid(out)

class ClassificationAttention(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationAttention, self).__init__()
        # self.fc = nn.Conv2d(num_classes, num_classes, 1, bias=True)
        self.fc = nn.Linear(num_classes, num_classes, bias = True)

    def forward(self, x):
        out = self.fc(x)
        return self.sigmoid(out)
