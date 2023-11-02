import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Resnet50Fc(nn.Module):
    def __init__(self):
        super(Resnet50Fc, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.__in_features = 256########

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        out = self.conv3(out)
        out = self.bn3(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        
        out = self.relu(out)
        x = out.view(out.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features




def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


