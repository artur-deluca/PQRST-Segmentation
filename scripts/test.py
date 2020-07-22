import torch
import numpy as np
from audicor_reader.reader import read_IEC
from evaluation.test_retinanet import test_retinanet
from utils.data_utils import IEC_dataset_preprocessing, onset_offset_generator, normalize
from utils.test_utils import qrs_seperation
import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

def test(net, signal_file_path):
    """
    test the signal using retinanet and return all of the predict result

    Args:
        net: (nn.Module) RetinaNet module
        signal_file_path: (string) this file must be .raw format
    Returns:
        intervals: (list) p duration, pq interval, qrs duration, qt interval mean and var value
        qrs_intervals: (list) q duration, r duration, and s duration mean value
    """
    """load signal file from .raw"""
    raw_ekg_sig = []
    try:
        sig = read_IEC(signal_file_path)
        sig = np.reshape(sig[0], (len(sig[0]), 1))
        raw_ekg_sig.append(sig.astype(float))
    except IOError:
        print(f"file {signal_file_path} does not exist.")

    ekg_sig = IEC_dataset_preprocessing(raw_ekg_sig, smooth=False, dns=False).to('cuda')
    denoise_sig = IEC_dataset_preprocessing(raw_ekg_sig, smooth=False, dns=True).to('cuda')

    """predict the pqrst segmentation result"""
    final_intervals = []
    final_preds = []
    denoise_sig = normalize(denoise_sig)
    for i in range(denoise_sig.size(0) // 128 + 1):
        plots, intervals, pred_signals = test_retinanet(net, denoise_sig[i*128:(i+1)*128, :, :], 4992, visual=False)
        print(intervals)
        final_intervals.extend(intervals)
        final_preds.append(pred_signals)
    final_preds = torch.cat(final_preds, dim=0)

    pred = qrs_seperation(ekg_sig, final_preds)

    for i in range(len(pred)):
        pred[i]["q_duration"] = np.mean(pred[i]["q_duration"])
        pred[i]["r_duration"] = np.mean(pred[i]["r_duration"])
        pred[i]["s_duration"] = np.mean(pred[i]["s_duration"])
    print(pred)

    return final_intervals, pred

def compare_to_standard(intervals, qrs_intervals, standard_intervals, standard_qrs_intervals):
    """
    compare the predict answer to standard and return mean difference value

    Args:
        intervals: (list) p duration, pq interval, qrs duration, qt interval mean and var value
        qrs_intervals: (list) q duration, r duration, and s duration mean value
        standard_intervals: (list) as intervals' structure
        standard_qrs_intervals: (list) as qrs_intervals' structure
    Returns:
        mean_diff: (Array) with sized [#signals, 4] represent the signals' 4 segments mean difference
        qrs_mean_diff: (Array) with sized [#signals, 3] represent the qrs segmets' mean difference
    """
    mean_diff = np.zeros((4, len(intervals)))
    for i in range(len(intervals)):
        mean_diff[0][i] = intervals[i]["p_duration"]["mean"] - standard_intervals[i]["p_duration"]["mean"]
        mean_diff[1][i] = intervals[i]["pq_interval"]["mean"] - standard_intervals[i]["pq_interval"]["mean"]
        mean_diff[2][i] = intervals[i]["qrs_duration"]["mean"] - standard_intervals[i]["qrs_duration"]["mean"]
        mean_diff[3][i] = intervals[i]["qt_interval"]["mean"] - standard_intervals[i]["qt_interval"]["mean"]

    qrs_mean_diff = np.zeros((3, len(qrs_intervals)))
    for i in range(len(qrs_intervals)):
        qrs_mean_diff[0][i] = qrs_intervals[i]["q_duration"]*2 - standard_qrs_intervals[i]["q_duration"]
        qrs_mean_diff[1][i] = qrs_intervals[i]["r_duration"]*2 - standard_qrs_intervals[i]["r_duration"]
        qrs_mean_diff[2][i] = qrs_intervals[i]["s_duration"]*2 - standard_qrs_intervals[i]["s_duration"]

    return mean_diff, qrs_mean_diff