import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch

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

    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)

    return X, Y

def get_mask(table, length):
    mask = [0] * length
    for triplet in table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end, 1):
            mask[i] = 1
    return mask

def get_background(p, qrs, t):
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i]==0 and qrs[i]==0 and t[i]==0:
            background[i]=1
    return background

def load_dataset(raw_dataset=raw_dataset_path, leads_seperate=True):
    X, Y = load_raw_dataset(raw_dataset)
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

    # (num_input, points, 4 labels)
    Y = np.repeat(Y, repeats=12, axis=0)
    # (num_input * 12, points, 4 labels)
    Y = np.swapaxes(Y, 1, 2)
    # (num_input * 12, 4 labels, points)

    X = torch.Tensor(X)
    Y = torch.Tensor(Y)

    return Data.TensorDataset(X, Y)


if __name__ == "__main__":
    xy = load_dataset()
