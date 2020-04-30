import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import viz
import matplotlib.pyplot as plt
import pickle
import wandb
wandb.init(project="PQRST-segmentation")

from model import UNet
from data_generator import dataset_preprocessing
from audicor_reader.reader import read_IEC
from utils.val_utils import validation_duration_accuracy
from utils.data_utils import onset_offset_generator, signal_get_mask, onset_offset_unsmooth_and_combine

def test(net, x, ground_truth=None):
    net.eval()
    # input size should be (num_of_signals, 1, 500 * seconds)
    with torch.no_grad():
        output = net(x)
    # output size should be (num_of_signals, 6, 500 * seconds)
    output = onset_offset_unsmooth_and_combine(output)
    output = signal_get_mask(output)
    if ground_truth is not None:
        ground_truth = onset_offset_unsmooth_and_combine(ground_truth)
        ground_truth = signal_get_mask(ground_truth)
        plot = viz.predict_plotter(x[0][0], output[0], ground_truth[0])
    else:
        plot = viz.predict_plotter(x[0][0], output[0])

    #pred_ans = F.one_hot(output.argmax(1), num_classes=4).permute(0, 2, 1)

    #output_onset_offset = onset_offset_generator(pred_ans[:, :3, :])
    output_onset_offset = onset_offset_generator(output)
    intervals = validation_duration_accuracy(output_onset_offset)
    return plot, intervals

def test_using_IEC():
    tol = 30
    # (num of ekg signal, length, 1)
    ekg_sig = []
    for i in range(1, 126):
        ekg_filename = '/home/Wr1t3R/PQRST/unet/data/IEC/IEC_from_audicor/CSE'+ str(i).rjust(3, '0') + '.raw'
        try:
            sig = read_IEC(ekg_filename)
            sig = np.reshape(sig[0], (len(sig[0]), 1))
            ekg_sig.append(sig)
        except IOError:
            print("file {} does not exist".format("CSE"+str(i).rjust(3, '0')))
    ekg_sig = dataset_preprocessing(ekg_sig, smooth=False)
    ekg_sig = ekg_sig.to('cuda')
    net = UNet(in_ch=1, out_ch=6)
    net.to('cuda')
    net.load_state_dict(torch.load("model.pkl"))
    plot, intervals = test(net, ekg_sig)

    table = []
    for i in range(len(intervals)):
        temp = [i, intervals[i]["p_duration"]["mean"], intervals[i]["pq_interval"]["mean"], intervals[i]["qrs_duration"]["mean"], intervals[i]["qt_interval"]["mean"]]
        table.append(temp)

    wandb.log({'viz': plot})
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
    print(correct / total)


if __name__ == "__main__":
    test_using_IEC()
