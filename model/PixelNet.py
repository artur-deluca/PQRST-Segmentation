import torch
import torch.nn as nn
from torch.autograd import Variable

from model.resnet_FPN import ResNet18_FPN

class PixelNet(nn.Module):
    def __init__(self, num_classes=3, out_length=4992):
        super(PixelNet, self).__init__()
        self.fpn = ResNet18_FPN()
        self.num_classes = num_classes
        self.linear = nn.Linear(1170, out_length)
        self.predict_layer = self._make_predict(num_classes + 1)

    def forward(self, x):
        fms = self.fpn(x)
        fm_concat = torch.cat(fms, dim=-1)
        out = self.linear(fm_concat)
        out = self.predict_layer(out)
        return out
    
    def _make_predict(self, out_plane):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv1d(256, out_plane, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
