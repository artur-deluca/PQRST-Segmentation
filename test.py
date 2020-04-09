import numpy as np
import torch
import viz
import matplotlib.pyplot as plt

def test(net, x, y):
    net.eval()
    # input size should be (1, 1, 500 * seconds)
    output = net(x)
    # output size should be (1, 4, 500 * seconds)
    plot = viz.predict_plotter(x[0][0], output[0], y[0])

    return plot
