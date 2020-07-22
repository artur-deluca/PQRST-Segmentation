import numpy as np
import torch
from data.encoder import DataEncoder
from utils.data_utils import box_to_sig_generator, onset_offset_generator
from utils.viz_utils import predict_plotter
import wandb
import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

def validation_accuracy(pred_onset_offset, gt_onset_offset):
    """
    The evaluation method that are used in the paper "Deep Learning for ECG Segmentation"

    Args:
        pred_onset_offset:  (Tensor) with sized [batch_size, #channels, signal_length]
        gt_onset_offset:    (Tensor) with sized [batch_size, #channels, signal_length]

    Returns:
        TP: (int) True positive count
        FP: (int) False positive count
        FN: (int) False negetive count
    """
    # (batch_size, 4, seconds) only first 3 channels will be used

    tol = int(config["General"]["F1_tolerance"])

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
    """
    the evaluation method that are used in IEC dataset

    Args:
        onset_offset: (Tensor) with sized [batch_size, #channels=4, signal length]
    
    Returns:
        output: (list of dict) with sized [batch_size], see below for detail dict structure
    """
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
    # test data should be (data_size, means and vars)
    output = []
    # Note: if there's missing segments then the value should be... 0?
    for i in range(one.shape[0]):
        p_duration = []
        pq_interval = []
        qrs_duration = []
        qt_interval = []
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
                    temp_pq += 1
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
                    temp_qt += 1
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
    return output

def eval_unet(net, loader, device):
    """
    the evaluation function that can be used during UNet training.

    Args:
        net:    (nn.Module) UNet module variable
        loader: (DataLoader) validation dataloader
        device: (str) using GPU or CPU
    Returns:
        average loss:   (float) average loss within validation set
        pointwise acc:  (float) pointwise evaluation
        Se:             (float) TP / (TP + FN)
        PPV:            (float) TP / (TP + FP)
        F1:             (float) 2 * Se * PPV / (Se + PPV)
        ret:            (list of dict) see validation_duration_accuracy for more detail
    """
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

            # only use first 3 channels because first three channels will produce all onsets/offsets
            pred_onset_offset = onset_offset_generator(pred_ans[:, :3, :])
            gt_onset_offset = onset_offset_generator(ground_truth[:, :3, :])
            tp, fp, fn = validation_accuracy(pred_onset_offset, gt_onset_offset)
            ret = validation_duration_accuracy(pred_onset_offset)
            TP += tp
            FP += fp
            FN += fn

            pbar.update()

    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * Se * PPV / (Se + PPV)

    return tot / n_val, correct / total, Se, PPV, F1, ret


def eval_retinanet(model, dataloader):
    """
    the evaluation function that can be used during RetinaNet training.

    Args:
        model:      (nn.Module) RetinaNet module variable
        dataloader: (DataLoader) validation dataloader
        
    Returns:
        Se:     (float) TP / (TP + FN)
        PPV:    (float) TP / (TP + FP)
        F1:     (float) 2 * Se * PPV / (Se + PPV)
    """
    input_length = 3968
    model.eval()
    
    pred_sigs = []
    gt_sigs = []
    sigs = []
    for batch_idx, (inputs, loc_targets, cls_targets, gt_boxes, gt_labels, gt_peaks) in enumerate(dataloader):
        batch_size = inputs.size(0)
        inputs = torch.autograd.Variable(inputs.cuda())
        loc_targets = torch.autograd.Variable(loc_targets.cuda())
        cls_targets = torch.autograd.Variable(cls_targets.cuda())
        inputs = inputs.unsqueeze(1)
        sigs.append(inputs)

        loc_preds, cls_preds = model(inputs)

        loc_preds = loc_preds.data.squeeze().type(torch.FloatTensor) # sized [#anchors * 3, 2]
        cls_preds = cls_preds.data.squeeze().type(torch.FloatTensor) # sized [#ahchors * 3, 3]

        loc_targets = loc_targets.data.squeeze().type(torch.FloatTensor)
        cls_targets = cls_targets.data.squeeze().type(torch.LongTensor)

        # decoder only process data 1 by 1.
        encoder = DataEncoder()
        for i in range(batch_size):
            boxes, labels, sco, is_found = encoder.decode(loc_preds[i], cls_preds[i], input_length)

        #ground truth decode using another method
            gt_boxes_tensor = torch.tensor(gt_boxes[i])
            gt_labels_tensor = torch.tensor(gt_labels[i])
            xmin = gt_boxes_tensor[:, 0].clamp(min=1)
            xmax = gt_boxes_tensor[:, 1].clamp(max=input_length - 1)
            gt_sig = box_to_sig_generator(xmin, xmax, gt_labels_tensor, input_length, background=False)
            
            if is_found:
                boxes = boxes.ceil()
                xmin = boxes[:, 0].clamp(min = 1)
                xmax = boxes[:, 1].clamp(max = input_length - 1)

                # there is no background anchor on predict labels
                pred_sig = box_to_sig_generator(xmin, xmax, labels, input_length, background=False)
            else:
                pred_sig = torch.zeros(1, 4, input_length)

            pred_sigs.append(pred_sig)
            gt_sigs.append(gt_sig)
    sigs = torch.cat(sigs, 0)
    pred_signals = torch.cat(pred_sigs, 0)
    gt_signals = torch.cat(gt_sigs, 0)
    plot = predict_plotter(sigs[0][0], pred_signals[0], gt_signals[0])
    #wandb.log({"visualization": plot})
    pred_onset_offset = onset_offset_generator(pred_signals)
    gt_onset_offset = onset_offset_generator(gt_signals)
    TP, FP, FN = validation_accuracy(pred_onset_offset, gt_onset_offset)

    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * Se * PPV / (Se + PPV)

    print("Se: {} PPV: {} F1 score: {}".format(Se, PPV, F1))
    wandb.log({"Se": Se, "PPV": PPV, "F1": F1})
    
    return Se, PPV, F1