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
    Args:
        net:            (nn.Module) RetinaNet model
        x:              (Tensor) with sized [#signals, 1 lead, values]
        input_length:   (int) input length must dividable by 64
        ground_truth:   (Tensor) with sized [batch_size, #anchors, 2]

    Returns:
        plot:       (pyplot) pyplot object
        interval:   (list of dict) with sized [#signals], for more info about dict structure, you can see utils.val_utils.validation_duration_accuracy.
    """
    net.eval()
    loc_preds, cls_preds = net(x)
    
    loc_preds = loc_preds.data.type(torch.FloatTensor)
    cls_preds = cls_preds.data.type(torch.FloatTensor)

    if ground_truth:
        loc_targets, cls_targets = ground_truth
        loc_targets = loc_targets.data.type(torch.FloatTensor)
        cls_targets = cls_targets.data.type(torch.LongTensor)
    
    batch_size = x.size(0)
    encoder = DataEncoder()

    pred_sigs = []
    gt_sigs = []
    for i in range(batch_size):
        boxes, labels, sco, is_found = encoder.decode(loc_preds[i], cls_preds[i], input_length, CLS_THRESH=0.425, NMS_THRESH=0.5)
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
    return plot, intervals, pred_signals

def test_retinanet_using_IEC(net, visual=False):
    """
    Args:
        net: (nn.Module) retinanet model variable.
    Returns:
        result: (list) with sized [4]. IEC standard accuracy evaluate using retinanet.
    """
    tol_pd = 10
    tol_pri = 10
    tol_qrsd = 10
    tol_qt = 25
    tol_std_pd = 15
    tol_std_pri = 10
    tol_std_qrsd = 10
    tol_std_qt = 30
    
    ekg_sig = load_IEC(denoise=wandb.config.test_denoise, pre=True)
    #ekg_sig = torch.nn.ConstantPad1d(15, 0)(ekg_sig)[:, :, :4992]

    plot, intervals, _ = test_retinanet(net, ekg_sig, 4992, visual=visual)

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
    mean_diff_ans = np.zeros((4, len(intervals)))
    for i in range(len(intervals)):
        mean_diff_ans[0][i] = table_mean[i][1] - df["P-duration"][i]
        mean_diff_ans[1][i] = table_mean[i][2] - df["PQ-interval"][i]
        mean_diff_ans[2][i] = table_mean[i][3] - df["QRS-duration"][i]
        mean_diff_ans[3][i] = table_mean[i][4] - df["QT-interval"][i]

        """count the percentage that can pass the tolerance"""
        if abs(mean_diff_ans[0][i]) <= tol_pd:# and table_var[i][1] <= tol_std_pd ** 2:
            correct[0] += 1
        if abs(mean_diff_ans[1][i]) <= tol_pri:# and table_var[i][2] <= tol_std_pri ** 2:
            correct[1] += 1
        if abs(mean_diff_ans[2][i]) <= tol_qrsd:# and table_var[i][3] <= tol_std_qrsd ** 2:
            correct[2] += 1
        if abs(mean_diff_ans[3][i]) <= tol_qt:# and table_var[i][4] <= tol_std_qt ** 2:
            correct[3] += 1
        total += 1

    mean_diff_ans = removeworst8(mean_diff_ans)

    mean_mean_diff = mean_diff_ans.mean(axis=1)
    std_mean_diff = mean_diff_ans.std(axis=1)
    ans = ["Fail!", "Fail!", "Fail!", "Fail!"]
    if abs(mean_mean_diff[0]) <= tol_pd and std_mean_diff[0] <= tol_std_pd:
        ans[0] = "Passed"
    if abs(mean_mean_diff[1]) <= tol_pri and std_mean_diff[1] <= tol_std_pri:
        ans[1] = "Passed"
    if abs(mean_mean_diff[2]) <= tol_qrsd and std_mean_diff[2] <= tol_std_qrsd:
        ans[2] = "Passed"
    if abs(mean_mean_diff[3]) <= tol_qt and std_mean_diff[3] <= tol_std_qt:
        ans[3] = "Passed"
    
    print(mean_mean_diff)
    print(std_mean_diff)
    print(ans)

    wandb.log({"pd_mean_diff_mean": mean_mean_diff[0], 
                "pri_mean_diff_mean": mean_mean_diff[1], 
                "qrsd_mean_diff_mean": mean_mean_diff[2], 
                "qt_mean_diff_mean": mean_mean_diff[3]})
    wandb.log({"pd_mean_diff_std": std_mean_diff[0],
                "pri_mean_diff_std": std_mean_diff[1],
                "qrsd_mean_diff_std": std_mean_diff[2],
                "qt_mean_diff_mean": std_mean_diff[3]})
    
    result = correct/total
    wandb.log({"result_pd": result[0], "result_pri": result[1], "result_qrsd": result[2], "result_qt": result[3]})
    return result, ans

def removeworst8(mean_diff):
    mean_diff = np.swapaxes(mean_diff, 0, 1)
    mean_diff = (mean_diff[np.abs(mean_diff[:, 1]).argsort()[::-1]])[8:,:]
    return mean_diff.swapaxes(0, 1)