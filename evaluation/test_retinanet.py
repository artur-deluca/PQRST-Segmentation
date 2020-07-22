import numpy as np
import pandas as pd
import torch
import wandb

from utils.viz_utils import predict_plotter
from model.RetinaNet import RetinaNet
from utils.val_utils import validation_duration_accuracy
from utils.data_utils import onset_offset_generator, box_to_sig_generator, one_hot_embedding, normalize
from utils.test_utils import load_IEC, load_ANE_CAL, get_signals_turning_point_by_rdp, enlarge_qrs_list, find_index_closest_to_value, removeworst
from data.encoder import DataEncoder

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

def test_retinanet(net, x, input_length, ground_truth=None, visual=False):
    """
    test the RetinaNet by any preprocessed signals.

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
    load IEC dataset and preprocess the signals, then testing RetinaNet using test_reitnanet function

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
    #table_var = []
    for i in range(len(intervals)):
        temp = [i, intervals[i]["p_duration"]["mean"], intervals[i]["pq_interval"]["mean"], intervals[i]["qrs_duration"]["mean"], intervals[i]["qt_interval"]["mean"]]
        table_mean.append(temp)
        #temp = [i, intervals[i]["p_duration"]["var"], intervals[i]["pq_interval"]["var"], intervals[i]["qrs_duration"]["var"], intervals[i]["qt_interval"]["var"]]
        #table_var.append(temp)

    wandb.log({'visualization': plot})
    wandb.log({"table_mean": wandb.Table(data=table_mean, columns=["file_name", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])})
    #wandb.log({"table_var": wandb.Table(data=table_var, columns=["file_name", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])})

    correct = np.zeros(4)
    total = np.zeros(4)
    df = pd.read_excel(config["General"]["CSE_label_path"], sheet_name=1, header=1)
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
    result_df = pd.DataFrame(mean_diff_ans.swapaxes(0, 1), columns=["p_duration", "pq_interval", "qrs_duration", "qt_interval"])
    mean_result_df = pd.DataFrame(table_mean, columns=["index", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])
    mean_result_df.to_excel(r'./mean_result.xlsx', header=True)
    result_df.to_excel(r'./result.xlsx', header=True)
    mean_diff_ans = removeworst(mean_diff_ans, 8)

    mean_mean_diff = mean_diff_ans.mean(axis=1)
    std_mean_diff = mean_diff_ans.std(axis=1, ddof=1)
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

def test_retinanet_using_ANE_CAL(net, visual=False):
    """
    load ANE and CAL dataset and preprocess the signals, testing the result using test_retinanet function

    Args:
        net: (nn.Module) retinanet model variable.
    Returns:
        result: (list) with sized [4]. IEC standard accuracy evaluate using retinanet.
    """
    tol_pd = 10
    tol_pri = 10
    tol_qrsd = 6
    tol_qt = 12
    tol_std_pd = 8
    tol_std_pri = 8
    tol_std_qrsd = 5
    tol_std_qt = 10
    
    ekg_sig = load_ANE_CAL(denoise=True, pre=True)
    #ekg_sig = torch.nn.ConstantPad1d(15, 0)(ekg_sig)[:, :, :4992]

    plot, intervals, _ = test_retinanet(net, ekg_sig, 4992, visual=visual)


    table_mean = []
    for i in range(len(intervals)):
        temp = [i, intervals[i]["p_duration"]["mean"], intervals[i]["pq_interval"]["mean"], intervals[i]["qrs_duration"]["mean"], intervals[i]["qt_interval"]["mean"]]
        table_mean.append(temp)

    wandb.log({'visualization': plot})
    wandb.log({"table_mean": wandb.Table(data=table_mean, columns=["file_name", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])})
    table_mean = np.reshape(np.array(table_mean), (-1, 5, 5)).mean(axis=1)

    correct = np.zeros(4)
    total = np.zeros(4)
    df = pd.read_excel(config["General"]["CAL_label_path"], sheet_name=0, header=0)
    mean_diff_ans = np.zeros((4, table_mean.shape[0]))
    for i in range(table_mean.shape[0]):
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
    result_df = pd.DataFrame(mean_diff_ans.swapaxes(0, 1), columns=["p_duration", "pq_interval", "qrs_duration", "qt_interval"])
    mean_result_df = pd.DataFrame(table_mean, columns=["index", "p_duration", "pq_interval", "qrs_duration", "qt_interval"])
    mean_result_df.to_excel(r'./mean_result_CAL.xlsx', header=True)
    result_df.to_excel(r'./result_CAL.xlsx', header=True)
    mean_diff_ans = removeworst(mean_diff_ans, 4)

    mean_mean_diff = mean_diff_ans.mean(axis=1)
    std_mean_diff = mean_diff_ans.std(axis=1, ddof=1)
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


def test_retinanet_by_qrs(net):
    """
    testing the CAL and ANE dataset q, r, s duration using rdp algorithm.

    Args:
        net: (nn.Module) Retinanet module
    """
    ekg_sig = load_ANE_CAL(denoise=False, pre=False, nor=False)
    turn_point = get_signals_turning_point_by_rdp(ekg_sig, load=False)
    print(len(turn_point[0]))
    
    final_preds = []
    ekg_sig = normalize(ekg_sig)
    for i in range(ekg_sig.size(0) // 128 + 1):
        _, _, pred_signals = test_retinanet(net, ekg_sig[i*128:(i+1)*128, :, :], 4992, visual=False)
        final_preds.append(pred_signals)
    final_preds = torch.cat(final_preds, dim=0)
    ekg_sig = ekg_sig.cpu().numpy()

    onset_offset = onset_offset_generator(final_preds)
    qrs_interval = []
    for i in range(onset_offset.shape[0]):
        qrs_interval.append([])
        j = 0
        while j < 4992:
            if onset_offset[i, 2, j] == -1:
                qrs_interval[i].append([j])
                j += 1
                while onset_offset[i, 2, j] == 0:
                    j += 1
                qrs_interval[i][-1].append(j)
            j += 1
    
    enlarge_qrs = enlarge_qrs_list(qrs_interval)

    turning = []
    for index in range(ekg_sig.shape[0]):
        turning.append([])
        for j in range(len(enlarge_qrs[index])):
            filtered_peaks = list(filter(lambda i: i >= enlarge_qrs[index][j][0] and i <= enlarge_qrs[index][j][1], turn_point[index]))
            turning[index].append(filtered_peaks)
            idx = find_index_closest_to_value(ekg_sig[index, 0, filtered_peaks[1]:filtered_peaks[2]], ekg_sig[index, 0, filtered_peaks[0]])
            idx = idx + filtered_peaks[1] - enlarge_qrs[index][j][0]
    
    pred = []
    for i in range(len(turning)):
        pred.append({"q_duration": [], "r_duration": [], "s_duration": []})
        mode = np.argmax(np.bincount([len(i) for i in turning[i]]))
        for j in range(len(turning[i])):
            if len(turning[i][j]) != mode:
                continue
            if mode >= 5:
                # q,r,s
                # find q duration
                q_end = find_index_closest_to_value(ekg_sig[i, 0, turning[i][j][1]: turning[i][j][2]], ekg_sig[i, 0, turning[i][j][0]])
                q_end = q_end + turning[i][j][1]
                q_duration = q_end - turning[i][j][0]
                pred[i]["q_duration"].append(q_duration)
                # find s duration
                s_start = find_index_closest_to_value(ekg_sig[i, 0, turning[i][j][2]: turning[i][j][3]], ekg_sig[i, 0, turning[i][j][4]])
                s_start = s_start + turning[i][j][2]
                s_duration = turning[i][j][4] - s_start
                pred[i]["s_duration"].append(s_duration)
                # find r duration
                r_start = q_end
                r_end = s_start
                r_duration = r_end - r_start
                pred[i]["r_duration"].append(r_duration)
            elif mode == 4:
                # q,r or r,s
                if ekg_sig[i, 0, turning[i][j][1]] > ekg_sig[i, 0, turning[i][j][2]]:
                    pred[i]["q_duration"].append(0)
                    # r, s            
                    # find s duration
                    s_start = find_index_closest_to_value(ekg_sig[i, 0, turning[i][j][1]: turning[i][j][2]], ekg_sig[i, 0, turning[i][j][3]])
                    s_start = s_start + turning[i][j][1]
                    s_duration = turning[i][j][3] - s_start
                    pred[i]["s_duration"].append(s_duration)
                    # find r duration
                    r_end = s_start
                    r_duration = r_end - turning[i][j][0]
                    pred[i]["r_duration"].append(r_duration)
                else:
                    if i == 84:
                        print(turning[i][j][1], turning[i][j][2])
                    # q, r
                    pred[i]["s_duration"].append(0)
                    # find q duration
                    q_end = find_index_closest_to_value(ekg_sig[i, 0, turning[i][j][1]: turning[i][j][2]], ekg_sig[i, 0, turning[i][j][0]])
                    q_end = q_end + turning[i][j][1]
                    q_duration = q_end - turning[i][j][0]
                    pred[i]["q_duration"].append(q_duration)                
                    # find r duration
                    r_start = q_end
                    r_duration = turning[i][j][3] - r_start
                    pred[i]["r_duration"].append(r_duration)
            elif mode <= 3:
                # only q or r
                if ekg_sig[i, 0, turning[i][j][1]] > ekg_sig[i, 0, turning[i][j][0]]:
                    # r
                    pred[i]["q_duration"].append(0)
                    pred[i]["s_duration"].append(0)
                    r_duration = turning[i][j][2] - turning[i][j][0]
                    pred[i]["r_duration"].append(r_duration)
                else:
                    # q
                    pred[i]["r_duration"].append(0)
                    pred[i]["s_duration"].append(0)
                    q_duration = turning[i][j][2] - turning[i][j][0]
                    pred[i]["q_duration"].append(q_duration)

    standard_qrs = []
    # ANE
    standard_qrs.append({"q_duration": 12, "r_duration": 52, "s_duration": 30})
    standard_qrs.append({"q_duration": 12, "r_duration": 52, "s_duration": 30})
    standard_qrs.append({"q_duration": 12, "r_duration": 52, "s_duration": 30})
    #CAL
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})
    standard_qrs.append({"q_duration": 0, "r_duration": 56, "s_duration": 0})
    standard_qrs.append({"q_duration": 0, "r_duration": 56, "s_duration": 0})
    standard_qrs.append({"q_duration": 0, "r_duration": 56, "s_duration": 0})
    standard_qrs.append({"q_duration": 56, "r_duration": 0, "s_duration": 0})
    standard_qrs.append({"q_duration": 56, "r_duration": 0, "s_duration": 0})
    standard_qrs.append({"q_duration": 56, "r_duration": 0, "s_duration": 0})
    standard_qrs.append({"q_duration": 0, "r_duration": 18, "s_duration": 18})
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})
    standard_qrs.append({"q_duration": 0, "r_duration": 50, "s_duration": 50})

    mean_diff = np.zeros((3, 17))
    for i in range(17):
        q_temp_mean = []
        r_temp_mean = []
        s_temp_mean = []
        for j in range(5):
            q_temp_mean.append(np.mean(pred[i*5+j]["q_duration"]))
            r_temp_mean.append(np.mean(pred[i*5+j]["r_duration"]))
            s_temp_mean.append(np.mean(pred[i*5+j]["s_duration"]))
        mean_diff[0][i] = np.mean(q_temp_mean)*2 - standard_qrs[i]["q_duration"]
        mean_diff[1][i] = np.mean(r_temp_mean)*2 - standard_qrs[i]["r_duration"]
        mean_diff[2][i] = np.mean(s_temp_mean)*2 - standard_qrs[i]["s_duration"]
    print(pd.DataFrame(mean_diff.T, columns=["q","r","s"]))
    print(np.mean(mean_diff, axis=1))
    print(np.std(mean_diff, axis=1, ddof=1))
    mean_diff = removeworst(mean_diff, 4)
    mean_diff_mean = np.mean(mean_diff, axis=1)
    mean_diff_std = np.std(mean_diff, axis=1, ddof=1)
    print(mean_diff_mean)
    print(mean_diff_std)