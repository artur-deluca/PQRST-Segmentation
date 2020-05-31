import torch
import numpy as np
import pandas as pd

from model.UNet import UNet
from utils.data_utils import onset_offset_generator
from utils.val_utils import validation_duration_accuracy
from utils.test_utils import load_IEC
from utils.viz_utils import predict_plotter


def test(net, x, ground_truth=None):
    net.eval()
    # input size should be (num_of_signals, 1, 500 * seconds)
    with torch.no_grad():
        output = net(x)
    # output size should be (num_of_signals, 4, 500 * seconds)
    if ground_truth is not None:
        plot = predict_plotter(x[0][0], output[0], ground_truth[0])
    else:
        plot = predict_plotter(x[0][0], output[0])

    pred_ans = F.one_hot(output.argmax(1), num_classes=4).permute(0, 2, 1)

    output_onset_offset = onset_offset_generator(pred_ans[:, :3, :])
    intervals = validation_duration_accuracy(output_onset_offset)
    return plot, intervals

def test_using_IEC(net):
    tol = 30    
    ekg_sig = load_IEC(pre=True)
    plot, intervals = test(net, ekg_sig)

    table = []
    for i in range(len(intervals)):
        temp = [i, intervals[i]["p_duration"]["mean"], intervals[i]["pq_interval"]["mean"], intervals[i]["qrs_duration"]["mean"], intervals[i]["qt_interval"]["mean"]]
        table.append(temp)

    wandb.log({'visualization': plot})
    wandb.log({"table": wandb.Table(data=table, columns=["file_name", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])})

    correct = np.zeros(4)
    total = np.zeros(4)
    df = pd.read_excel("/home/Wr1t3R/PQRST/unet/data/CSE_Multilead_Library_Interval_Measurements_Reference_Values.xls", sheet_name=1, header=1)
    for i in range(len(intervals)):
        if abs(table[i][1] - df["P-duration"][i]) < tol:
            correct[0] += 1
        if abs(table[i][2] - df["PQ-interval"][i]) < tol:
            correct[1] += 1
        if abs(table[i][3] - df["QRS-duration"][i]) < tol:
            correct[2] += 1
        if abs(table[i][4] - df["QT-interval"][i]) < tol:
            correct[3] += 1
        total += 1

    result = correct/total
    wandb.log({"result_pd": result[0], "result_pri": result[1], "result_qrsd": result[2], "result_qt": result[3]})
    return result