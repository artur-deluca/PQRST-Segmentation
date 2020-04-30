import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import BaselineWanderRemoval as bwr
from tqdm import tqdm

import viz
from utils.data_utils import smooth_signal, smooth_label

import torch.utils.data as Data
leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 500
raw_dataset_path="/home/Wr1t3R/PQRST/unet/data/ecg_data_200.json"

def load_raw_dataset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    Y = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'])

        signal_len = 5000
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
        p_delin = delineation_tables['p']
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p = get_mask(p_delin, signal_len)
        qrs = get_mask(qrs_delin, signal_len)
        t = get_mask(t_delin, signal_len)
        background = get_background(p, qrs, t)

        y.append(p)
        y.append(qrs)
        y.append(t)
        y.append(background)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    # ()
    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)

    return X, Y

def load_raw_dataset_with_onset_offset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    Y = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'])

        signal_len = 5000
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
        p_delin = delineation_tables['p']
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p_onset, p_offset = get_edge_onset_offset(p_delin, signal_len)
        qrs_onset, qrs_offset = get_edge_onset_offset(qrs_delin, signal_len)
        t_onset, t_offset = get_edge_onset_offset(t_delin, signal_len)

        y.append(p_onset)
        y.append(p_offset)
        y.append(qrs_onset)
        y.append(qrs_offset)
        y.append(t_onset)
        y.append(t_offset)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    # X(number of signals, 12 leads, signal)
    X = np.swapaxes(X, 1, 2)
    # X(number of signals, signal, 12 leads)
    # Y(number of signals, 6 labels, signal)
    Y = np.swapaxes(Y, 1, 2)
    # Y(number of signals, signal, 6 labels)

    return X, Y

def get_mask(table, length):
    mask = [0] * length
    for triplet in table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end, 1):
            mask[i] = 1
    return mask

def get_edge_onset_offset(table, length):
    onset = [0] * length
    offset = [0] * length
    for triplet in table:
        onset[triplet[0]] = 1
        offset[triplet[2]] = 1
    return onset, offset

def get_background(p, qrs, t):
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i]==0 and qrs[i]==0 and t[i]==0:
            background[i]=1
    return background

def load_dataset(raw_dataset=raw_dataset_path, leads_seperate=True, fix_baseline_wander=False, smooth_input=True, smooth_target=True, onset_offset=True):
    if onset_offset:
        X, Y = load_raw_dataset_with_onset_offset(raw_dataset)
    else:
        X, Y = load_raw_dataset(raw_dataset)

    if fix_baseline_wander:
        X = baselineWanderRemoval(X, FREQUENCY_OF_DATASET)

    if smooth_input:
        smoothed = []
        # number of signals
        for i in range(X.shape[0]):
            # leads
            for j in range(X.shape[2]):
                X[i, :, j] = smooth_signal(X[i, :, j])[:X.shape[1]]

    if smooth_target:
        Y = Y.astype(np.float32)
        # number of signals
        for i in range(Y.shape[0]):
            # labels
            for j in range(Y.shape[2]):
                Y[i, :, j] = smooth_label(Y[i, :, j], window_len=11)

    # data augmentation and modification
    # delete first and last 2 seconds
    X, Y = X[:, 1000:4000, :], Y[:, 1000:4000, :]
    # data augmentation by randomly choosing 4 seconds to load to the model
    X = np.concatenate((np.concatenate((X[:, 0:2000, :], X[:, 500:2500, :]), axis=0), X[:, 1000:3000, :]), axis=0)
    Y = np.concatenate((np.concatenate((Y[:, 0:2000, :], Y[:, 500:2500, :]), axis=0), Y[:, 1000:3000, :]), axis=0)

    if leads_seperate == True:
        # (num_input, points, 12 leads)
        X = np.swapaxes(X, 1, 2)
        # (num_input, 12 leads, points)
        X = np.reshape(X, (X.shape[0] * X.shape[1], 1, X.shape[2]))
        # (num_input * 12, 1, points)

    # (num_input, points, 6 labels)
    Y = np.repeat(Y, repeats=12, axis=0)
    # (num_input * 12, points, 6 labels)
    Y = np.swapaxes(Y, 1, 2)
    # (num_input * 12, 6 labels, points)

    X = torch.Tensor(X)
    Y = torch.Tensor(Y)

    return Data.TensorDataset(X, Y)

def baselineWanderRemoval(signal, frequency):
    num_data = signal.shape[0]
    num_leads = signal.shape[2]
    for i in tqdm(range(num_data)):
        for j in range(num_leads):
            signal[i, :, j] = bwr.fix_baseline_wander(signal[i, :, j], frequency)
    return signal

# this function is used to preprocess the IEC signal
def dataset_preprocessing(data, leads_seperate=True, smooth=False):
    # (# of data, points, leads)
    data = np.swapaxes(data, 1, 2)
    # (# of data, leads, points)
    if leads_seperate == True:
        if data.shape[1] > 2:
            data = np.reshape(data, (data.shape[0] * data.shape[1], 1, data.shape[2]))

    if smooth:
        smoothed = []
        for i in range(data.shape[0]):
            smoothed.append(smooth_signal(data[i, 0, :]))

        smoothed = np.array(smoothed)[:, np.newaxis, :5008]
        smoothed = torch.Tensor(smoothed)

        return smoothed

    else:
        #data = data[:, :, 500:4500]
        data = data[:, :, :4992]
        data = torch.Tensor(data)

        return data

def preview_data():
    X, Y = load_raw_dataset_with_onset_offset(raw_dataset=raw_dataset_path)
    smoothed = []
    for i in range(Y.shape[2]):
        # lead 2
        smoothed.append(smooth_label(Y[0, :, i], window_len=11))
    viz.signals_plot_all(np.array(smoothed), name="label_smooth")

if __name__ == "__main__":
    load_dataset()
