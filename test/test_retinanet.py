import numpy as np
import pandas as pd
import torch
import wandb

from utils.viz_utils import predict_plotter
from model.RetinaNet import RetinaNet
from utils.val_utils import validation_duration_accuracy
from utils.data_utils import onset_offset_generator, box_to_sig_generator, one_hot_embedding
from utils.test_utils import load_IEC
from data.encoder import DataEncoder


def test_retinanet(net, x, input_length, ground_truth=None, visual=False):
    """
    Params:
        net: (nn.Module) RetinaNet model
        x: (Tensor) with sized [#signals, 1 lead, values]
        input_length: (int) input length must dividable by 64
        ground_truth: (Tensor) with sized [batch_size, #anchors, 2]
    """
    net.eval()
    loc_preds, cls_preds = net(x)
    
    loc_preds = loc_preds.data.squeeze().type(torch.FloatTensor)
    cls_preds = cls_preds.data.squeeze().type(torch.FloatTensor)

    if ground_truth:
        loc_targets, cls_targets = ground_truth
        loc_targets = loc_targets.data.squeeze().type(torch.FloatTensor)
        cls_targets = cls_targets.data.squeeze().type(torch.LongTensor)
    
    batch_size = x.size(0)
    encoder = DataEncoder()

    pred_sigs = []
    gt_sigs = []
    for i in range(batch_size):
        boxes, labels, sco, is_found = encoder.decode(loc_preds[i], cls_preds[i], input_length, CLS_THRESH=0.5, NMS_THRESH=0.5)
        if is_found:
            boxes = boxes.ceil()
            xmin = boxes[:, 0].clamp(min = 1)
            xmax = boxes[:, 1].clamp(max = input_length - 1)
        
            pred_sig = box_to_sig_generator(xmin, xmax, labels, input_length, background=False)

        else:
            pred_sig = torch.zeros(1, 4, input_length)
        if ground_truth:
            gt_boxes, gt_labels, gt_sco, gt_is_found = encoder.decode(loc_targets[i], one_hot_embedding(cls_targets[i], 4), input_length)
            gt_sig = box_to_sig_generator(gt_boxes[:, 0], gt_boxes[:, 1], gt_labels, input_length, background=False)
            gt_sigs.append(gt_sig)
        pred_sigs.append(pred_sig)
    pred_signals = torch.cat(pred_sigs, 0)
    pred_onset_offset = onset_offset_generator(pred_signals)
    plot = None
    if visual:
        if ground_truth is not None:
            for i in range(batch_size):
                plot = predict_plotter(x[i][0], pred_signals[i], ground_truth[i], name=str(i))
        else:
            for i in range(batch_size):
                plot = predict_plotter(x[i][0], pred_signals[i], name=str(i))
        
    if ground_truth:
        gt_signals = torch.cat(gt_sigs, 0)
        gt_onset_offset = onset_offset_generator(gt_signals)
        TP, FP, FN = validation_accuracy(pred_onset_offset, gt_onset_offset)
    
    intervals = validation_duration_accuracy(pred_onset_offset[:, 1:, :])
    return plot, intervals

def test_retinanet_using_IEC(net):
    tol_pd = 10
    tol_pri = 10
    tol_qrsd = 6
    tol_qt = 12
    tol_std_pd = 8
    tol_std_pri = 8
    tol_std_qrsd = 5
    tol_std_qt = 10
    
    ekg_sig = load_IEC(pre=True)

    plot, intervals = test_retinanet(net, ekg_sig, 4992, visual=False)

    table_mean = []
    table_var = []
    for i in range(len(intervals)):
        temp = [i, intervals[i]["p_duration"]["mean"], intervals[i]["pq_interval"]["mean"], intervals[i]["qrs_duration"]["mean"], intervals[i]["qt_interval"]["mean"]]
        table_mean.append(temp)
        temp = [i, intervals[i]["p_duration"]["var"], intervals[i]["pq_interval"]["var"], intervals[i]["qrs_duration"]["var"], intervals[i]["qt_interval"]["var"]]
        table_var.append(temp)

    wandb.log({'visualization': plot})
    wandb.log({"table_mean": wandb.Table(data=table_mean, columns=["file_name", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])})
    wandb.log({"table_var": wandb.Table(data=table_var, columns=["file_name", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])})

    correct = np.zeros(4)
    total = np.zeros(4)
    df = pd.read_excel("/home/Wr1t3R/PQRST/unet/data/CSE_Multilead_Library_Interval_Measurements_Reference_Values.xls", sheet_name=1, header=1)
    for i in range(len(intervals)):
        if abs(table_mean[i][1] - df["P-duration"][i]) <= tol_pd:# and table_var[i][1] <= tol_std_pd ** 2:
            correct[0] += 1
        if abs(table_mean[i][2] - df["PQ-interval"][i]) <= tol_pri:# and table_var[i][2] <= tol_std_pri ** 2:
            correct[1] += 1
        if abs(table_mean[i][3] - df["QRS-duration"][i]) <= tol_qrsd:# and table_var[i][3] <= tol_std_qrsd ** 2:
            correct[2] += 1
        if abs(table_mean[i][4] - df["QT-interval"][i]) <= tol_qt:# and table_var[i][4] <= tol_std_qt ** 2:
            correct[3] += 1
        total += 1
    result = correct/total
    wandb.log({"result_pd": result[0], "result_pri": result[1], "result_qrsd": result[2], "result_qt": result[3]})
    return result