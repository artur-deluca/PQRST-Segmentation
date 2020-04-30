import numpy as np
import torch

def validation_accuracy(pred_onset_offset, gt_onset_offset):
    # if (batch_size, 4, seconds) only first 3 channels will be used

    tol = 10

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

def validation_duration_accuracy(onset_offset):
    # (batch_size, 4, seconds) only first 3 channels will be used.
    # need to merge 3 segments to 1
    onset_offset[:,1,:] *= 2
    onset_offset[:,2,:] *= 3
    one = np.sum(onset_offset, axis=1)
    # shape become (batch_size, seconds) ?
    # if there is no lapping on each segment then this should work
    # finish label merging
    """
    p duration: between -1 and 1
    pq interval: between -1 and -2
    qrs duration: between -2 and 2
    qt interval: between -2 and 3
    """
    p_duration = []
    pq_interval = []
    qrs_duration = []
    qt_interval = []

    # test data should be (data_size, means and vars)
    output = []
    # Note: if there's missing segments then the value should be... 0?
    for i in range(one.shape[0]):
        j = 0
        while j < one.shape[1]:
            temp_p = 0
            temp_pq = 0
            temp_qrs = 0
            temp_qt = 0
            if j < one.shape[1] and one[i, j] == -1:
                j += 1
                while j < one.shape[1] and one[i, j] == 0:
                    j += 1
                    temp_p += 1
                    temp_pq += 1
                if j < one.shape[1] and one[i, j] == 1:
                    p_duration.append(temp_p * 2)
                    j += 1
                while j < one.shape[1] and one[i, j] == 0:
                    j += 1
                    temp_pq += 1
                if j < one.shape[1] and one[i, j] == -2:
                    pq_interval.append(temp_pq * 2)
                temp_p = 0
                temp_pq = 0
            if j < one.shape[1] and one[i, j] == -2:
                j += 1
                while j < one.shape[1] and one[i, j] == 0:
                    temp_qrs += 1
                    temp_qt += 1
                    j += 1
                if j < one.shape[1] and one[i, j] == 2:
                    qrs_duration.append(temp_qrs * 2)
                    j += 1
                # -2...2...-3...3
                while j < one.shape[1] and (one[i, j] == 0 or one[i, j] == -3):
                    temp_qt += 1
                    j += 1
                if j < one.shape[1] and one[i, j] == 3:
                    qt_interval.append(temp_qt * 2)
                temp_qrs = 0
                temp_qt = 0
            j += 1
        ret = {
            "p_duration": {
                "mean": np.mean(p_duration),
                "var": np.var(p_duration)
            },
            "pq_interval": {
            "mean": np.mean(pq_interval),
            "var": np.var(pq_interval)
            },
            "qrs_duration": {
                "mean": np.mean(qrs_duration),
                "var": np.var(qrs_duration)
            },
            "qt_interval": {
                "mean": np.mean(qt_interval),
                "var": np.var(qt_interval)
            }
        }
        output.append(ret)
        p_duration = []
        pq_interval = []
        qrs_duraiton = []
        qt_interval = []
    return output
