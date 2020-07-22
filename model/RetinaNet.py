import os, sys
import torch
import torch.nn as nn

from model.resnet_FPN import ResNet18_FPN


class RetinaNet(nn.Module):
    """
    the whole model of RetinaNet
    """
    # 3 scales and this is 1d, only need 1 for each scales
    num_anchors = 3

    # include background label
    def __init__(self, num_classes=3):
        super(RetinaNet, self).__init__()
        self.fpn = ResNet18_FPN()
        self.num_classes = num_classes
        # ctr + width, or start + width
        self.loc_head = self._make_head(self.num_anchors * 2)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        # feature maps
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # (N, 3 * 2, L) -> (N, L, 3 * 2) -> (N, L * 3, 2)
            loc_pred = loc_pred.permute(0, 2, 1).contiguous().view(x.size(0), -1, 2)
            # (N, 3 * 4, L) -> (N, L, 3 * 4) -> (N, L * 3, 4)
            cls_pred = cls_pred.permute(0, 2, 1).contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        return loc_preds, cls_preds

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv1d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
