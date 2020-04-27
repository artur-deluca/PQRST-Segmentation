import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection

def predict_plotter(x, pred, y=None):
    x = x.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    color = ["red", "blue", "green", "black"]
    color_arg_pred = np.argmax(pred, axis=0)
    color_table_pred = []

    if y is not None:
        y = y.detach().cpu().numpy()
        color_arg_y = np.argmax(y, axis=0)
        color_table_y = []
        for t in range(x.shape[0]):
            color_table_y.append(color[color_arg_y[t]])

    for t in range(x.shape[0]):
        color_table_pred.append(color[color_arg_pred[t]])

    lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(range(x.shape[0])[:-1], x[:-1], range(x.shape[0])[1:], x[1:])]

    if y is not None:
        fig, axes = plt.subplots(2, 1, figsize=(40, 5))
        ax1 = axes[0]
        ax2 = axes[1]
        ax1.set_title('predict')
        for index, line in enumerate(lines):
            xx, yy = zip(line[0], line[1])
            ax1.plot(xx, yy, color=color_table_pred[index])
        #ax1.scatter(range(x.shape[0]), x, c=color_table_pred, s=1)
        ax2.set_title('ground truth')
        for index, line in enumerate(lines):
            xx, yy = zip(line[0], line[1])
            ax2.plot(xx, yy, color=color_table_y[index])

    else:
        fig, ax1 = plt.subplots(1,1, figsize=(40, 5))
        ax1.set_title('predict')
        for index, line in enumerate(lines):
            xx, yy = zip(line[0], line[1])
            ax1.plot(xx, yy, color=color_table_pred[index])

    plt.savefig("./out_pic/test.png")

    return plt

def signals_plot_all(signals):
    # signals(number_of_signals, signal)
    for i in range(signals.shape[0]):
        plt.close('all')
        plt.plot(range(signals.shape[1]), signals[i])
        plt.savefig("./out_pic/" + str(i) + ".png")
    plt.close('all')