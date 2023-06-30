import torch.nn as nn
import torch


def autopad(k):
    return k // 2

class Inception_blockV1(nn.Module):
    def __init__(self, in_channel, out_channel=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, autopad(3), bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 5, 1, autopad(5), bias=False)
        self.conv3 = nn.Conv2d(in_channel, out_channel, 7, 1, autopad(7), bias=False)
        self.conv4 = nn.Conv2d(in_channel, out_channel, 9, 1, autopad(9), bias=False)
        self.bn = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat([out1, out2, out3, out4], 1)  # (b, 16*4, 224, 224)
        out = self.relu(self.bn(out))
        out = self.maxpool(out)  # (b, 16*4, 112, 112)
        return out

# /4
class Inception_blockV2(nn.Module):
    def __init__(self, in_channel=16, out_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 2, bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 2, bias=False), 
                                   nn.Conv2d(out_channel, out_channel, 3, 1, autopad(3), bias=False))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 2, bias=False), 
                                   nn.Conv2d(out_channel, out_channel, 5, 1, autopad(5), bias=False))
        self.conv4 = nn.Sequential(nn.MaxPool2d(3, 2, 1), 
                                   nn.Conv2d(in_channel, out_channel, 5, 1, autopad(5), bias=False))
        self.bn = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat([out1, out2, out3, out4], 1)  # (b, 16*4, 224, 224)
        out = self.relu(self.bn(out))
        out = self.maxpool(out)  # (b, 16*4, 112, 112)
        return out
    
class Basic_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, autopad(3), bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Embedding_layer(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_channel, 256, bias=False), 
                                 nn.BatchNorm1d(256), 
                                 nn.ReLU(True))
        
        self.fc2 = nn.Sequential(nn.Linear(256, 128, bias=False), 
                                 nn.BatchNorm1d(128), 
                                 nn.ReLU(True))
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class Inception(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.inception_blockV1 = Inception_blockV1(in_channel, 16)
        self.inception_blockV2 = Inception_blockV2(64, 32)
        self.basic_block1 =Basic_block(128, 256)
        self.basic_block2 =Basic_block(256, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding_layer = Embedding_layer(512)
        self.norm = nn.Identity()
    
    def forward(self, x):
        x = self.inception_blockV1(x)#  (b, 16*4, 112, 112)
        x = self.inception_blockV2(x)# (b, 128, 28, 28)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.embedding_layer(x)
        x = self.norm(x)
        return x

if __name__ == '__main__':
    inp = torch.randn(1, 3, 224, 224)
    m = Inception(3)
    out = m(inp)
    print(out.shape)
