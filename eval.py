import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)
    tot = 0
    correct = 0
    total = 0

    TP = 0
    FP = 0
    FN = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, ncols=100) as pbar:
        for batch in loader:
            x, ground_truth = batch[0], batch[1]
            x = x.to(device, dtype=torch.float32)
            ground_truth = ground_truth.to(device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(x)

            tot += F.binary_cross_entropy_with_logits(pred, ground_truth).item()
            # (batch_size, channels, data)
            pred_ans = F.one_hot(pred.argmax(1), num_classes=4).permute(0, 2, 1)
            correct += pred_ans.eq(ground_truth).sum().item()
            total += ground_truth.shape[0] * ground_truth.shape[1] * ground_truth.shape[2]

            pred_onset_offset = onset_offset_generator(pred_ans)
            gt_onset_offset = onset_offset_generator(ground_truth)
            tp, fp, fn = validation_accuracy(pred_onset_offset, gt_onset_offset)
            TP += tp
            FP += fp
            FN += fn
            
            pbar.update()
        
    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * Se * PPV / (Se + PPV)
        
    return tot / n_val, correct / total, Se, PPV, F1
   

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
