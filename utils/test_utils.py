import torch
import numpy as np
import json
from utils.data_utils import normalize, IEC_dataset_preprocessing, signal_rdp, onset_offset_generator
from audicor_reader.reader import read_IEC
import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

def load_IEC(denoise=True, pre=False):
    """
    loading IEC CSE dataset and preprocess the signal.

    Arg:
        pre: (bool) load from saved preprocessed data or not
        denoise: (bool) preprocess using wavelet thresholding or not
    Returns:
        ekg_sig: (Tensor) with sized [#signals, 1 lead, signal_length]
    """
    # (num of ekg signal, length, 1)
    if pre:
        ekg_sig = torch.load(config["RetinaNet"]["output_path"]+"IEC_preprocessed_data.pt").to('cuda')
    else:
        ekg_sig = [] 
        for i in range(1, 126):
            ekg_filename = config["General"]["IEC_path"]+'CSE'+ str(i).rjust(3, '0') + '.raw'
            try:
                sig = read_IEC(ekg_filename)
                sig = np.reshape(sig[0], (len(sig[0]), 1))
                ekg_sig.append(sig.astype(float))
            except IOError:
                print("file {} does not exist".format("CSE"+str(i).rjust(3, '0')))
        
        ekg_sig = IEC_dataset_preprocessing(ekg_sig, smooth=False, dns=denoise)
        ekg_sig = ekg_sig.to('cuda')
        ekg_sig = normalize(ekg_sig, instance=True)
        torch.save(ekg_sig, config["RetinaNet"]["output_path"]+"IEC_preprocessed_data.pt")

    return ekg_sig

def load_ANE_CAL(denoise=True, pre=False, save=True, nor=True):
    """
    loading IEC ANE and CAL dataset and preprocess the signal

    Arg:
        denoise: (bool) denoise using wavelet threshold or not
        pre: (bool) load from saved preprocessed data or not
        save: (bool) save preprocessed data to file or not
        normalize: (bool) normalize the signal or not
    Return:
        ekg_sig: (Tensor) with shape [#signals, 1 lead, signal_length]
    """
    name = json.loads(config["General"]["CAL_name"])
    # (num of ekg signal, length, 1)
    if pre:
        ekg_sig = torch.load(config["RetinaNet"]["output_path"]+"CAL_preprocessed_data.pt").to('cuda')
    else:
        ekg_sig = []
        for i in range(len(name)):
            for j in range(1, 6):
                ekg_filename = f'{config["General"]["IEC_path"]}{name[i]}_{str(j)}.raw'
                try:
                    sig = read_IEC(ekg_filename)
                    sig = np.reshape(sig[0], (len(sig[0]), 1))
                    ekg_sig.append(sig.astype(float))
                except IOError:
                    print(f"file {name[i]}_{str(j)} does not exist")
        
        ekg_sig = IEC_dataset_preprocessing(ekg_sig, smooth=False, dns=denoise)
        
        ekg_sig = ekg_sig.to('cuda')
        if nor:
            ekg_sig = normalize(ekg_sig, instance=True)
        if save:
            torch.save(ekg_sig, config["RetinaNet"]["output_path"]+"CAL_preprocessed_data.pt")

    return ekg_sig

def get_signals_turning_point_by_rdp(signals, load=True, save=True):
    """
    get the signals' turning points by rdp algorithm

    Args:
        signals: (Tensor) with sized [#signals, 1 lead, signal_length]
        load: (bool) to load preprocessed turning point to data or not
        save: (bool) save this to pt or not
    Returns:
        turn_point: (list) with sized [#signals, #turning_points_per_signal]
    """
    if load:
        turn_point = torch.load(config["RetinaNet"]["output_path"]+"CAL_turning_point_preprocessed_data.pt")
        return turn_point
    turn_point = []
    signals = signals.cpu().numpy()
    for i in range(signals.shape[0]):
        ret = signal_rdp(signals[i][0], 7.5)
        turn_point.append(ret)
    if save:
        torch.save(turn_point, config["RetinaNet"]["output_path"]+"CAL_turning_point_preprocessed_data.pt")
    
    return turn_point

def enlarge_qrs_list(origin):
    """
    enlarge the qrs complex range to make sure it contains all q, r, s waves

    Args:
        origin: (list) with sized [#signals, #qrs_duration_detected]
    Return:
        ret: (list) with sized [#signals, #qrs_duration_detected]
    """
    ret = origin.copy()
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            ret[i][j][0] -= 7
            ret[i][j][1] += 7
    
    return ret

def find_index_closest_to_value(li, value):
    """
    to find a closest value in a given list

    Args:
        li: (list)
        value: (int)
    Returns:
        (int) closest value index in the list
    """
    li = np.array(li)
    return np.argmin(abs(li-value))

def removeworst(mean_diff, remove_num):
    """
    remove the worst results from mean_diff and return the result

    Args:
        mean_diff: (Array) with sized [#segments, #signals]
        remove_num: (int) #signals to be removed
    Returns:
        mean_diff: (Array) with sized [#segments, #singals - remove_num] that remove the worst remove_num number of data
    """
    mean_diff = np.take_along_axis(mean_diff, np.abs(mean_diff).argsort(axis=1), axis=1)[:, :-remove_num]
    return mean_diff

def qrs_seperation(ekg_sig, final_preds):
    turn_point = get_signals_turning_point_by_rdp(ekg_sig, load=False)
    ekg_sig = ekg_sig.cpu().numpy()

    """qrs segmentation"""
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
    return pred