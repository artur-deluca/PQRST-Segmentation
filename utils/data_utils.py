import numpy as np
import torch

def onset_offset_generator(sig):
    """
    :signal type: tensor
    :return type: numpy array
    """
    sig = sig.cpu().numpy()
    # sig(batch_size, 4, seconds)
    # next signal value
    next_sig = np.roll(sig, -1, axis=2)

    # onset will be -1 and offset will be 1 (background channel is useless)
    onset_offset = sig - next_sig

    return onset_offset


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