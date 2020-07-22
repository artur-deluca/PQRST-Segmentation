import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    The basic block that are used in resnet18/34
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        """
        basic block with 2 conv layers.
        first layer will downsample the size to inputsize/2 (using stride) and increase the features channels, last layer will not change the input size and features channels from the output of first layer.
        """
        self.conv1 = nn.Conv1d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=planes)
        
        """SE"""
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_down = nn.Conv1d(planes, planes // 16, kernel_size=1, bias=False)
        self.conv_up = nn.Conv1d(planes // 16, planes, kernel_size=1, bias=False)

        self.linear_down = nn.Linear(planes, planes // 16, bias=False)
        self.linear_up = nn.Linear(planes // 16, planes, bias=False)

        self.sigmoid = nn.Sigmoid()
        
        """
        residual link
        """
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(num_features=self.expansion * planes)
            )
    
    def forward(self, x):
        """basic block"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        """SE"""
        out1 = self.global_pool(out).squeeze()
        out1 = self.linear_down(out1)
        out1 = self.relu(out1)
        out1 = self.linear_up(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.unsqueeze(-1)
        
        """residual"""
        out = out * out1.expand_as(out) + self.downsample(x)
        #out += self.downsample(x)
        out = self.relu(out)
        
        return out
        

class FPN(nn.Module):
    """
    the RetinaNet structure. backbone using resnet18, and connect with a simple FPN that remove the p7 layer.
    """
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        # first layer of resnet
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        # resnet
        self.layer1 = self._make_layer(block, planes=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, num_blocks=num_blocks[3], stride=2)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)

        # lateral layers
        self.lateral_layer1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.lateral_layer2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.lateral_layer3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding=0)
        self.lateral_layer4 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1, padding=0)

        # smooth layers
        self.smooth1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        # only first layer need to downsample input length
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        # batch, channel, rest
        _, _, length = y.size()

        return F.interpolate(x, size=(length), mode='nearest') + y
    
    def forward(self, x):
        # resnet
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool1d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN first layer
        p6 = self.conv6(c5)
        
        # FPN rest
        p5 = self.lateral_layer1(c5)
        p4 = self._upsample_add(p5, self.lateral_layer2(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.lateral_layer3(c3))
        p3 = self.smooth2(p3)

        return p3, p4, p5, p6

def ResNet18_FPN():
    """
    the entry point of this model
    """
    return FPN(BasicBlock, [2, 2, 2, 2])
