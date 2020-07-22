import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

class UNet(nn.Module):
    """a simple UNet from paper 'Deep Learning for ECG Segmentation'"""
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        # conv1d + batchnorm1d + relu
        self.conv1 = self.ConvNet(in_ch, 4, 9, 1, 4)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = self.ConvNet(4, 8, 9, 1, 4)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3 = self.ConvNet(8, 16, 9, 1, 4)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.conv4 = self.ConvNet(16, 32, 9, 1, 4)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        # bottle neck ( conv1d + batchnorm1d + relu)
        self.conv5 = self.ConvNet(32, 64, 9, 1, 4)
        # upconv1d
        self.upconv1 = self.ConvTransNet(64)
        self.conv6 = self.ConvNet(96, 32, 9, 1, 4)
        self.upconv2 = self.ConvTransNet(32)
        self.conv7 = self.ConvNet(48, 16, 9, 1, 4)
        self.upconv3 = self.ConvTransNet(16)
        self.conv8 = self.ConvNet(24, 8, 9, 1, 4)
        self.upconv4 = self.ConvTransNet(8)
        self.conv9 = self.ConvNet(12, 4, 9, 1, 4)
        self.final = nn.Conv1d(4, out_ch, 1)
        
        # upconv
    def ConvNet(self, in_ch, out_ch, kernel_size, stride, padding):
        net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
                )
        return net

    def ConvTransNet(self, ch):
        net = nn.ConvTranspose1d(ch, ch, 8, stride=2, padding=3) 
        # in_ch should equal to out_ch in this paper
        return net
    
    def forward(self, x):
        c1 = self.conv1(x)
        mp1 = self.pool1(c1)
        c2 = self.conv2(mp1)
        mp2 = self.pool2(c2)
        c3 = self.conv3(mp2)
        mp3 = self.pool3(c3)
        c4 = self.conv4(mp3)
        mp4 = self.pool4(c4)
        c5 = self.conv5(mp4)
        up6 = self.upconv1(c5)
        cat6 = torch.cat((c4, up6), dim=1)
        c6 = self.conv6(cat6)
        up7 = self.upconv2(c6)
        cat7 = torch.cat((c3, up7), dim=1)
        c7 = self.conv7(cat7)
        up8 = self.upconv3(c7)
        cat8 = torch.cat((c2, up8), dim=1)
        c8 = self.conv8(cat8)
        up9 = self.upconv4(c8)
        cat9 = torch.cat((c1, up9), dim=1)
        c9 = self.conv9(cat9)
        f = self.final(c9)
        return f

