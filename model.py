import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# optimized weighted binary cross entropy loss
class Loss(nn.modules.Module):
    def __init__(self, Wt):
        super(Loss, self).__init__()
        self.Wt = Wt

    def forward(self, inputs, targets, cat):
        loss = -(self.Wt[cat]['Wt1'] * targets * inputs.log() + self.Wt[cat]['Wt0'] * (1 - targets) * (
                    1 - inputs).log())
        #         print (loss)
        #         print (loss.sum())
        return loss.mean()


class PretrainedDensenet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)
        for params in densenet_169.parameters():
            params.requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=4)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        features = self.features(x)
        out = self.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)
        return self.sigmoid(self.fc1(out))


class PretrainedDensenet_one(nn.Module):
    def __init__(self, num_class=1):
        super(PretrainedDensenet_one,self).__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)
        for params in densenet_169.parameters():
            params.requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)   # kernel_size->3
        self.features = nn.Sequential(*list(densenet_169.features.children()))
#         self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        features = self.features(x)
#         out = self.relu(features)
        out = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = out.view(-1, self.channels)
        return self.sigmoid(self.fc1(out))


# 以下transform全部为一个通道输入
class PretrainedDensenet_two(nn.Module):
    def __init__(self, num_class=1):
        super(PretrainedDensenet_two,self).__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)
        for params in densenet_169.parameters():
            params.requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        out = self.sigmoid(self.fc1(out))
        return out



class PretrainedDensenet_three(nn.Module):
    def __init__(self, num_class=1):
        super(PretrainedDensenet_three,self).__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)
        for params in densenet_169.parameters():
            params.requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        features = self.features(x)
        out = self.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)
        return self.sigmoid(self.fc1(out))