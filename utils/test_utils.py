import torch
import numpy as np
from utils.data_utils import normalize, IEC_dataset_preprocessing, signal_rdp
from audicor_reader.reader import read_IEC

def load_IEC(denoise=True, pre=False):
    """
    Arg:
        pre: (bool) load from saved preprocessed data or not
    """
    # (num of ekg signal, length, 1)
    if pre:
        ekg_sig = torch.load("./data/IEC_preprocessed_data.pt").to('cuda')
    else:
        ekg_sig = []
        for i in range(1, 126):
            ekg_filename = '/home/Wr1t3R/PQRST/unet/data/IEC/IEC_from_audicor/CSE'+ str(i).rjust(3, '0') + '.raw'
            try:
                sig = read_IEC(ekg_filename)
                sig = np.reshape(sig[0], (len(sig[0]), 1))
                ekg_sig.append(sig.astype(float))
            except IOError:
                print("file {} does not exist".format("CSE"+str(i).rjust(3, '0')))
        
        ekg_sig = IEC_dataset_preprocessing(ekg_sig, smooth=False, dns=denoise)
        ekg_sig = ekg_sig.to('cuda')
        ekg_sig = normalize(ekg_sig, instance=True)
        torch.save(ekg_sig, "./data/IEC_preprocessed_data.pt")

    return ekg_sig

def load_ANE_CAL(denoise=True, pre=False, save=True, normalize=True):
    """
    Arg:
        denoise: (bool) denoise using wavelet threshold or not
        pre: (bool) load from saved preprocessed data or not
        save: (bool) save preprocessed data to file or not
    Return:
        ekg_sig: (Tensor) with shape [#signals, 1 lead, signal_length]
    """
    name = ["ANE20000", "ANE20001", "ANE20002", 
    "CAL05000", "CAL10000", "CAL15000", 
    "CAL20000", "CAL20002", "CAL20100", 
    "CAL20110", "CAL20160", "CAL20200", 
    "CAL20210", "CAL20260", "CAL20500",
    "CAL30000", "CAL50000"]
    # (num of ekg signal, length, 1)
    if pre:
        ekg_sig = torch.load("./data/CAL_preprocessed_data.pt").to('cuda')
    else:
        ekg_sig = []
        for i in range(len(name)):
            for j in range(1, 6):
                ekg_filename = f'/home/Wr1t3R/PQRST/unet/data/IEC/IEC_from_audicor/{name[i]}_{str(j)}.raw'
                try:
                    sig = read_IEC(ekg_filename)
                    sig = np.reshape(sig[0], (len(sig[0]), 1))
                    ekg_sig.append(sig.astype(float))
                except IOError:
                    print(f"file {name[i]}_{str(j)} does not exist")
        
        ekg_sig = IEC_dataset_preprocessing(ekg_sig, smooth=False, dns=denoise)
        
        ekg_sig = ekg_sig.to('cuda')
        if normalize:
            ekg_sig = normalize(ekg_sig, instance=True)
        if save:
            torch.save(ekg_sig, "./data/CAL_preprocessed_data.pt")

    return ekg_sig

def get_signals_turning_point_by_rdp(signals, load=True, save=True):
    """
    Args:
        signals: (Tensor) with sized [#signals, 1 lead, signal_length]
        load: (bool) to load preprocessed turning point to data or not
        save: (bool) save this to pt or not
    """
    if load:
        turn_point = torch.load("./data/CAL_turning_point_preprocessed_data.pt")
        return turn_point
    turn_point = []
    signals = signals.cpu().numpy()
    for i in range(signals.shape[0]):
        print(i)
        ret = signal_rdp(signals[i][0], 7.5)
        turn_point.append(ret)
    if save:
        torch.save(turn_point, "./data/CAL_turning_point_preprocessed_data.pt")
    
    return turn_point

def enlarge_qrs_list(origin):
    """
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
    Args:
        li: (list)
        value: (int)
    """
    li = np.array(li)
    return np.argmin(abs(li-value))

def removeworst(mean_diff, remove_num):
    """
    Args:
        mean_diff: (Array) with sized [#segments, #signals]
        remove_num: (int) #signals to be removed
    """
    mean_diff = np.take_along_axis(mean_diff, np.abs(mean_diff).argsort(axis=1), axis=1)[:, :-remove_num]
    return mean_diff