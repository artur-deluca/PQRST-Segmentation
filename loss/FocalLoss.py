import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.data_utils import one_hot_embedding

class FocalLoss(nn.Module):
    """
    the self-defined loss function 'focal loss' that are used in RetinaNet training
    """
    def __init__(self, num_classes=3):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        """
        Focal Loss

        Args:
        :param x: (tensor) sized (N, D)
        :param y: (tensor) sized (N, )
        :return :
                (tensor) focal loss
        """
        alpha = 0.25
        gamma = 2
        t = one_hot_embedding(y.data.cpu().long(), 1 + self.num_classes)
        t = t[:, 1:] # exclude background
        t = Variable(t).cuda()

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1- t)
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w.detach(), reduction='sum')

    def focal_loss_alt(self, x, y):
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu().long(), 1 + self.num_classes)
        t = t[:, 1:] # exclude background
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    @staticmethod
    def where(cond, x_1, x_2):
        return (cond.float() * x_1) + ((1 - cond.float()) * x_2)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """
        Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
            loc_preds: (tensor) predicted locations, sized (batch_size, #ahchors, 2)
            loc_targets: (tensor) encoded target locations, sized (batch_size, #anchors, 2)
            cls_preds: (tensor) predicted class confidences, sized (batch_size, #anchors, #classes)
            cls_targets: (tensor) encoded target labels, sized (batch_size, #anchors)

        Returns:
            (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets)
        """
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        """loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)"""
        if num_pos > 0:
            mask = pos.unsqueeze(2).expand_as(loc_preds) # (N, #anchors, 2)
            masked_loc_preds = loc_preds[mask].view(-1, 2) # (#pos, 2)
            masked_loc_targets = loc_targets[mask].view(-1, 2) # (#pos, 2)
            regression_diff = torch.abs(masked_loc_targets - masked_loc_preds)
            # this is sigma=3 l1 smooth loss
            loc_loss = self.where(torch.le(regression_diff, 1.0/9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), regression_diff - 0.5/9.0)

            loc_loss = loc_loss.mean()
        else:
            num_pos = 1.
            loc_loss = Variable(torch.Tensor([0]).float().cuda())
        
        """cls_loss = FocalLoss(cls_preds, cls_targets)"""
        pos_neg = cls_targets > -1 # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_prds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_prds, cls_targets[pos_neg])

        loc_loss = loc_loss
        cls_loss = cls_loss / num_pos
        return loc_loss, cls_loss