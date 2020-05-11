from model.RetinaNet import RetinaNet
import torch
import torch.nn as nn
from torch.autograd import Variable
# (batchsize, )
a = Variable(torch.randn(10, 1, 5000))
# (batch, number of anchors(which is like a segment(P, QRS, T)), 2 (means that ))
b = Variable(torch.randn(10, 30, 2))
c = Variable(torch.randn(10, 30))
net = RetinaNet()
loc, cls = net((a, b, c))
print(loc.size())
print(cls.size())