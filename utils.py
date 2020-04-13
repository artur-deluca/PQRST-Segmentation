import numpy as np
import torch

def onset_offset_generator(sig):
    """
    # sig(4, seconds)
    onset = np.zeros(sig.shape)
    offset = np.zeros(sig.shape)
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]-1):
            if sig[i][j] == 0 and sig[i][j+1] == 1:
                onset[i][j] = 1
            if sig[i][j] == 1 and sig[i][j+1] == 0:
                offset[i][j] = 1
    """
    sig = sig.cpu().numpy()
    # working
    # sig(batch_size, 4, seconds)
    # next signal value
    next_sig = np.roll(sig, -1, axis=2)
    
    # onset will be -1 and offset will be 1 (background channel is useless)
    onset_offset = sig - next_sig

    return onset_offset

def validation_accuracy(pred_onset_offset, gt_onset_offset):
    # (batch_size, 4, seconds) only first 3 channels will be used

    tol = 15
    
    TP = 0
    FP = 0
    FN = 0

    # np.where return struct: (dim1, dim2, dim3) where dims are array with same length.
    pred_onset_idx = np.argwhere(pred_onset_offset == -1)
    pred_offset_idx = np.argwhere(pred_onset_offset == 1)
    gt_onset_idx = np.argwhere(gt_onset_offset == -1)
    gt_offset_idx = np.argwhere(gt_onset_offset == 1)
    # record current point is used or not
    pred_onset_tag = np.where(pred_onset_offset == -1, 1, 0)
    pred_offset_tag = np.where(pred_onset_offset == 1, 1, 0)
    gt_onset_tag = np.where(gt_onset_offset == -1, 1, 0)
    gt_offset_tag = np.where(gt_onset_offset == 1, 1, 0)
    
    for i in range(len(pred_onset_idx)):
        # find ground truth have -1 or not on the -1 index of predicted array within tol range
        idx = np.argwhere(gt_onset_offset[pred_onset_idx[i][0]][pred_onset_idx[i][1]][pred_onset_idx[i][2]-tol:pred_onset_idx[i][2]+tol] == -1)
        # if find ground truth have -1 within tolerance range (0 if not find any so it won't get in this loop)
        for j in range(idx.size):
            if gt_onset_tag[pred_onset_idx[i][0]][pred_onset_idx[i][1]][pred_onset_idx[i][2]-tol+idx[j]] == 1:
                gt_onset_tag[pred_onset_idx[i][0]][pred_onset_idx[i][1]][pred_onset_idx[i][2]-tol+idx[j]] = 0
                pred_onset_tag[pred_onset_idx[i][0]][pred_onset_idx[i][1]][pred_onset_idx[i][2]] = 0
                TP += 1
                break

    for i in range(len(pred_offset_idx)):
        # find ground truth have -1 or not on the -1 index of predicted array within tol range
        idx = np.argwhere(gt_onset_offset[pred_offset_idx[i][0]][pred_offset_idx[i][1]][pred_offset_idx[i][2]-tol:pred_offset_idx[i][2]+tol] == 1)
        # if find ground truth have -1 within tolerance range (0 if not find any so it won't get in this loop)
        for j in range(idx.size):
            if gt_offset_tag[pred_offset_idx[i][0]][pred_offset_idx[i][1]][pred_offset_idx[i][2]-tol+idx[j]] == 1:
                gt_offset_tag[pred_offset_idx[i][0]][pred_offset_idx[i][1]][pred_offset_idx[i][2]-tol+idx[j]] = 0
                pred_offset_tag[pred_offset_idx[i][0]][pred_offset_idx[i][1]][pred_offset_idx[i][2]] = 0
                TP += 1
                break
    
    FP += np.sum(pred_onset_tag) + np.sum(pred_offset_tag)
    FN += np.sum(gt_onset_tag) + np.sum(gt_offset_tag)

    return TP, FP, FN
