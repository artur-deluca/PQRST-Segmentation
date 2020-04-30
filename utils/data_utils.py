import numpy as np
import torch
from scipy import signal
import sys
np.set_printoptions(threshold=sys.maxsize)

def onset_offset_generator(sig):
    """
    :signal type: tensor
    :return type: numpy array
    """
    try:
        sig = sig.cpu().numpy()
    except:
        print("signal is not a tensor")
    # sig(batch_size, 4, seconds)
    # next signal value
    next_sig = np.roll(sig, -1, axis=2)

    # onset will be -1 and offset will be 1 (background channel is useless)
    onset_offset = sig - next_sig

    return onset_offset

def onset_offset_unsmooth_and_combine(sig):
    """
    This function will unsmooth the signal to one-hot like array and combine onset and offset to form the input of the evaluate function
    """
    try:
        sig = sig.cpu().numpy()
    except:
        print("signal is not a tensor")
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            idx, _ = signal.find_peaks(sig[i, j], height=0.5, distance=30)
            sig[i, j] = np.zeros(sig.shape[2])
            for index in idx:
                sig[i, j, index] = 1
    
    # (num_of_signals, 3 labels, seconds)
    out = np.zeros((sig.shape[0], sig.shape[1] // 2, sig.shape[2]))
    # 0, 1, 2
    for i in range(out.shape[1]):
        # change onset to -1
        sig[:, 2 * i, :] = np.where(sig[:, i, :] == 1, -1, 0)
        out[:, i, :] = np.sum(sig[:, 2 * i: 2 * (i + 1), :], axis=1)

    return out
    

def smooth_signal(signal, window_len=20, window="hanning"):
    """
    :signal type: numpy array
    :window choise: "hanning", "flat", "hamming", "bartlett", "blackman"
    "return type: numpy array
    """
    if signal.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if signal.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size")
    
    if window_len < 3:
        return signal

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hamming', 'bartlett', 'blackman'")
    
    s=np.r_[signal[window_len-1:0:-1], signal, signal[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def smooth_label(label, window_len=1, window="gaussian"):
    if window_len == 1:
        return label
    label = label.astype(np.float32)
    if window=="gaussian":
        windows = signal.gaussian(window_len, 2)
    for i in range(len(label)):
        if label[i] == 1:
            label[i - (window_len - 1) // 2: i + (window_len + 1) // 2] = windows
    return label

def signal_get_mask(sig):
    # input should be the output of signal_onset_offset_unsmooth_and_combine function
    # (number_of_signals, 3, signal)
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            k = 0
            while k < sig.shape[2]:
                # onset
                if sig[i, j, k] == -1:
                    start = k
                    k += 1
                    while k < sig.shape[2] - 1 and sig[i, j, k] == 0:
                        k += 1
                    if sig[i, j, k] == 1:
                        end = k
                        sig[i, j, start:end] = np.ones(end-start)
                k += 1
    return sig