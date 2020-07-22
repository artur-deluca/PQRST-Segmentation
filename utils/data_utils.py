import numpy as np
import torch
import json
import pywt
from rdp import rdp
from scipy.signal import medfilt
from scipy.stats import entropy
import multiprocessing as mp
from audicor_reader.reader import read_IEC
import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 500
start_point = 516
end_point = 4484
raw_dataset_path = config["General"]["LUDB_path"]

def load_raw_dataset_and_bbox_labels(raw_dataset):
    """
    Get raw dataset with signals, bboxes and labels
    
    Args:
        raw_dataset: (string) dataset path
    
    Returns:
        X:      (Array) input signals with sized [#subjects, #leads, signal_length=5000]
        BBoxes: (list) target bboxes with sized [#subjects, #leads, #objs, 2]
        Labels: (list) target labels with sized [#subjects, #leads, #objs] 
    """
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    BBoxes = []
    Labels = []
    Peaks = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        bbox = []
        label = []
        peak = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]

            delineation_tables = leads[leads_names[0]]['DelineationDoc']
            
            p_delin = delineation_tables['p']
            qrs_delin = delineation_tables['qrs']
            t_delin = delineation_tables['t']

            # skip the data that missed at least one segment.
            if len(p_delin) == 0 or len(qrs_delin) == 0 or len(t_delin) == 0:
                continue

            # background label 0 will add when encoding
            p_boxes, p_labels, p_peaks = get_bbox_labels(p_delin, 0)
            qrs_boxes, qrs_labels, qrs_peaks = get_bbox_labels(qrs_delin, 1)
            t_boxes, t_labels, t_peaks = get_bbox_labels(t_delin, 2)
            
            b = [*p_boxes, *qrs_boxes, *t_boxes]
            l = [*p_labels, *qrs_labels, *t_labels]
            p = [*p_peaks, *qrs_peaks, *t_peaks]

            x.append(leads[lead_name]['Signal'][start_point:end_point])
            bbox.append(b)
            label.append(l)
            peak.append(p)

        X.append(x)
        BBoxes.append(bbox)
        Labels.append(label)
        Peaks.append(peak)

    return X, BBoxes, Labels, Peaks

def load_raw_dataset_and_bbox_labels_CAL():
    """
    Get raw dataset with signals, bboxes and labels
    
    Args:
        raw_dataset: (string) dataset path
    
    Returns:
        X:      (Array) input signals with sized [#subjects, #leads, signal_length=5000]
        BBoxes: (list) target bboxes with sized [#subjects, #leads, #objs, 2]
        Labels: (list) target labels with sized [#subjects, #leads, #objs] 
    """

    name = ["ANE20000", "ANE20001", "ANE20002",
     "CAL05000", "CAL10000", "CAL15000", 
     "CAL20000", "CAL20002", "CAL20100", 
     "CAL20110","CAL20160", "CAL20200", 
     "CAL20210", "CAL20260", "CAL20500", 
     "CAL30000", "CAL50000"]

    path = config["General"]["IEC_path"]
    
    X = []
    BBoxes = []
    Labels = []
    Peaks = []
    for i in range(len(name)):
        x = []
        bbox = []
        label = []
        peak = []
        for j in range(5):
            signal = list(read_IEC(path+name[i]+"_"+str(j+1)+".raw"))
            with open(path+"CAL_label/"+name[i]+"_"+str(j+1)+".json") as f:
                label_ = json.load(f)
            
            p_delin = label_['p']
            qrs_delin = label_['qrs']
            t_delin = label_['t']
            # background label 0 will add when encoding
            p_boxes, p_labels, p_peaks = get_bbox_labels(p_delin, 0)
            qrs_boxes, qrs_labels, qrs_peaks = get_bbox_labels(qrs_delin, 1)
            t_boxes, t_labels, t_peaks = get_bbox_labels(t_delin, 2)
            
            b = [*p_boxes, *qrs_boxes, *t_boxes]
            l = [*p_labels, *qrs_labels, *t_labels]
            p = [*p_peaks, *qrs_peaks, *t_peaks]

            x.append(signal[0][start_point:end_point])
            bbox.append(b)
            label.append(l)
            peak.append(p)

        X.append(x)
        BBoxes.append(bbox)
        Labels.append(label)
        Peaks.append(peak)

    return X, BBoxes, Labels, Peaks

def get_bbox_labels(delineation, label):
    """
    get bbox labels from delineation table

    Args: 
        delineation: (list) sized [#obj, 3], 3 indicates onset, peak, offset

    Returns:
        bboxes: (list) with sized [#obj, 2], 2 indicates (xmin, xmax)
        labels: (list) with sized [#obj]
    """
    bboxes = []
    labels = []
    peaks = []
    for obj in delineation:
        if len(obj) >= 3:
            xmin = obj[0]
            peak = obj[1]
            xmax = obj[2]
        elif len(obj) == 2:
            xmin = obj[0]
            xmax = obj[1]
            peak = (xmin + xmax) // 2
        if xmin >= start_point and xmax < end_point:
            bboxes.append((xmin - start_point, xmax - start_point))
            labels.append(label)
            peaks.append(peak)

    return bboxes, labels, peaks

def onset_offset_generator(sig):
    """
    get the onset and offset of p, qrs, t segmentation on signals

    Args:
        signal: (Tensor) with sized [batch_size, #channels, signal_length]

    Returns:
        onset_offset: (Array) with sized [batch_size, #channels, signal_length]
    """
    sig = sig.cpu().numpy()
    # sig(batch_size, 4, seconds)
    # next signal value
    next_sig = np.roll(sig, -1, axis=2)

    # onset will be -1 and offset will be 1 (background channel is useless)
    onset_offset = sig - next_sig

    return onset_offset

def box_to_sig_generator(xmin, xmax, labels, sig_length, background=True):
    """
    decode the bbox label to signal label.

    Args:
        xmin:       (Tensor) with sized [#obj]
        xmax:       (Tensor) with sized [#obj]
        labels:     (Tensor) with sized [#obj]
        sig_length: (int) signal length

    Returns:
        sig: (Tensor) with sized [1, 4, sig_length]
    """
    xmin = xmin.long()
    xmax = xmax.long()
    labels = labels.long()
    sig = torch.zeros(1, 4, sig_length)
    if background:
        for i in range(len(labels)):
            sig[0, labels[i], xmin[i]:xmax[i]] = 1
    else:
        for i in range(len(labels)):
            sig[0, labels[i]+1, xmin[i]:xmax[i]] = 1
    return sig

def smooth_signal(signal, window_len=20, window="hanning"):
    """
    smooth the signal by window averaging.

    Args:
        signal:     (Array) with sized [signal length]
        window_len: (int) window averaging's window size
        window:     (str) choise of window type: {"hanning", "flat", "hamming", "barlett", "blackman"}

    Returns:
        y: (Array) with sized [signal length]
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

def change_box_order(boxes, order):
    """
    change box order between (x_min, x_max) and (x_center, length)

    Args:
        boxes:  (Tensor) bounding boxes, sized [N, 2]
        order:  (str) either 'xx2xl' or 'xl2xx'
    
    Returns:
                (Tensor) converted bounding boxes, sized [N, 2]
    """
    assert order in ['xx2xl', 'xl2xx']
    a = boxes[:, :1]
    b = boxes[:, 1:]
    if order == 'xx2xl':
        return torch.cat([(a+b)/2, b-a+1], 1)
    return torch.cat([a-b/2, a+b/2], 1)

def box_iou(box1, box2, order='xx'):
    """
    Compute the intersection over union of two set of boxes

    The default box order is (xmin, xmax)

    Args:
        box1:   (Tensor) bounding boxes, sized [N, 2]
        box2:   (Tensor) bounding boxes, sized [M, 2]
        order:  (str) box order, either 'xx' or 'xl'
    
    Returns:
                (Tensor) iou, sized[N, M]
    """

    if order == 'xl':
        box1 = change_box_order(box1, 'xl2xx')
        box2 = change_box_order(box2, 'xl2xx')

    N = box1.size(0)
    M = box2.size(0)

    left = torch.max(box1[:, None, 0], box2[:, 0]) # [N, M]
    right = torch.min(box1[:, None, 1], box2[:, 1]) # [N, M]

    inter_length = (right - left + 1).clamp(min=0) # [N, M]

    left = torch.min(box1[:, None, 0], box2[:, 0])
    right = torch.max(box1[:, None, 1], box2[:, 1])

    union_length = (right - left + 1).clamp(min=0)

    return inter_length / union_length
    
def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    """
    Non maximum suppression

    Args:
        bboxes:     (Tensor) bounding boxes, sized [N, 2]
        scores:     (Tensor) bbox scores, szied [N, ]
        threshold:  (float) overlap threshold
        mode:       (str) 'union' or 'min'

    Returns:
        keep:   (Tensor) selected indices
        sco:    (Tensor) selected scores
    """
    x1 = bboxes[:, 0]
    x2 = bboxes[:, 1]

    areas = x2 - x1 + 1
    _, order = scores.sort(0, descending=True)
    keep = []
    sco = []
    while order.numel() > 0:
        i = order[0] if order.numel() > 1 else order.item()
        keep.append(i)
        sco.append(scores[i])

        if order.numel() == 1:
            break
        
        # clamp all start, end point within the highest score bbox range and calculate the intersection
        xx1 = x1[order[1:]].clamp(min=x1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])

        l = (xx2 - xx1 + 1).clamp(min=0)
        
        if mode == 'union':
            ovr = l / (areas[i] + areas[order[1:]] - l)
        elif mode == 'min':
            ovr = l / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)
        
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep), torch.Tensor(sco)

def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
        labels:         (LongTensor) class label, sized [N,].
        num_classes:    (int) number of classes.
    
    Returns:
                        (Tensor) encoded labels, size [N, #classes].
    """
    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]

def normalize(signals, instance=True):
    """
    normalize the signal.

    Args:
        signals:    (Tensor) with sized [batch_size, #leads, signal length]
        instance:   (bool) True means normalize using signals's own means and stds. False means using whole training data means and stds.
    
    Returns:
                    (Tensor) with sized [batch_size, #leads, signal length]
    """
    if instance:
        mean = signals.mean(-1).unsqueeze(-1).expand_as(signals)
        std = signals.std(-1).unsqueeze(-1).expand_as(signals)
        return (signals - mean) / std
    else:
        mean = signals.mean().unsqueeze(-1).unsqueeze(-1).expand_as(signals)
        std = signals.std().unsqueeze(-1).unsqueeze(-1).expand_as(signals)
        return (signals - mean) / std

def wavelet_threshold(data, wavelet='sym8', noiseSigma=14):
    """
    denoise the signal by wavelet thresholding.

    Args:
        data: (Array) with sized []
        wavelet:
        noiseSigma:
    """
    levels = int(np.floor(np.log2(data.shape[0])))
    WC = pywt.wavedec(data,wavelet,level=levels)
    threshold=noiseSigma*np.sqrt(2*np.log2(data.size))
    NWC = list(map(lambda x: pywt.threshold(x,threshold, mode='soft'), WC))
    return pywt.waverec(NWC, wavelet)

def baseline_wander_removal(data):
    """
    baseline wander removal.

    Args:
        data: (Array)
    """
    baseline = medfilt(data, 201)
    baseline = medfilt(baseline, 601)
    return data - baseline

def _denoise_mp(signal):
    """
    multi threading wavelet threshold denoise
    
    Args:
        signal: (Array)
    """
    return baseline_wander_removal(wavelet_threshold(signal))
    
def ekg_denoise(data, number_channels=None):
    '''
    Denoise the ekg data parallely and return.
    
    data: np.ndarray of shape [n_channels, n_samples]
    number_channels: the first N channels to be processed
    '''

    number_channels = data.shape[0] if number_channels is None else number_channels

    with mp.Pool(processes=number_channels) as workers:
        results = list()

        for i in range(number_channels):
            results.append(workers.apply_async(_denoise_mp, (data[i], )))

        workers.close()
        workers.join()

        for i, result in enumerate(results):
            data[i] = result.get()

    return data


def load_raw_dataset_and_pointwise_labels(raw_dataset):
    """
    load ludb dataset and return pointwise labels that are used for UNet.

    Args:
        raw_dataset: (str) raw dataset path
    Returns:
        X: (Array) with sized [#subjects, #leads, signal length]
        Y: (Array) with sized [#subjects, #channels=4, signal length]
    """
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    Y = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'])

        signal_len = 5000
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
        p_delin = delineation_tables['p']
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p = get_mask(p_delin, signal_len)
        qrs = get_mask(qrs_delin, signal_len)
        t = get_mask(t_delin, signal_len)
        background = get_background(p, qrs, t)

        y.append(p)
        y.append(qrs)
        y.append(t)
        y.append(background)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)

    return X, Y

def get_mask(table, length):
    """
    using label to mask signal

    Args:
        table: (list) with sized [#segments_in_signal, 3]
        length: (int) signal length
    Returns:
        mask: (list) with sized [length]
    """
    mask = [0] * length
    for triplet in table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end, 1):
            mask[i] = 1
    return mask

def get_background(p, qrs, t):
    """
    the background version of get_mask

    Args:
        p, qrs, t: (list) with sized [length], the masked signal
    """
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i]==0 and qrs[i]==0 and t[i]==0:
            background[i]=1
    return background


def load_dataset_using_pointwise_labels(raw_dataset=raw_dataset_path, leads_seperate=True, fix_baseline_wander=False, smooth=True):
    """
    load LUDB dataset and preprocess the data.

    Args:
        raw_dataset:            (str) raw dataset path
        leads_seperate:         (bool) True if all leads are look as different subjects
        fix_baseline_wander:    (bool) True if preprocessing using baseline wander removal
        smooth:                 (bool) True if preprocessing using window averaging smoothing method
    
    Returns:
                                (TensorDataset) with inputs and ground truth [X, Y]
    """

    X, Y = load_raw_dataset_and_pointwise_labels(raw_dataset)
    if smooth:
        smoothed = []
        # number of signals
        for i in range(X.shape[0]):
            # leads
            for j in range(X.shape[2]):
                smoothed.append(smooth_signal(X[i, :, j]))
        X = np.array(smoothed)[:, :5000, np.newaxis]
        

    # data augmentation and modification
    # delete first and last 2 seconds
    X, Y = X[:, 1000:4000, :], Y[:, 1000:4000, :]
    # data augmentation by randomly choosing 4 seconds to load to the model
    X = np.concatenate((np.concatenate((X[:, 0:2000, :], X[:, 500:2500, :]), axis=0), X[:, 1000:3000, :]), axis=0)
    Y = np.concatenate((np.concatenate((Y[:, 0:2000, :], Y[:, 500:2500, :]), axis=0), Y[:, 1000:3000, :]), axis=0)

    if leads_seperate == True:
        # (num_input, points, 12 leads)
        X = np.swapaxes(X, 1, 2)
        # (num_input, 12 leads, points)
        X = np.reshape(X, (X.shape[0] * X.shape[1], 1, X.shape[2]))
        # (num_input * 12, 1, points)

    # (num_input, points, 4 labels)
    Y = np.repeat(Y, repeats=12, axis=0)
    # (num_input * 12, points, 4 labels)
    Y = np.swapaxes(Y, 1, 2)
    # (num_input * 12, 4 labels, points)

    X = torch.Tensor(X)
    Y = torch.Tensor(Y)

    return Data.TensorDataset(X, Y)


def IEC_dataset_preprocessing(data, leads_seperate=True, smooth=False, dns=True):
    """
    preprocess the IEC signal.

    Args:
        data:           (Array) with sized [#signals, signal length, #leads]
        leads_seperate: (bool) True if all leads are look as different subjects' signals
        smooth:         (bool) True if preprocessing using window averaging smoothing method
        dns:            (bool) True if preprocessing using ekg denoise method
    
    Returns:
        data: (Tensor) with sized [#signals * #leads, 1, signal length] if leads_seperate==True
    """
    # (# of data, points, leads)
    data = np.swapaxes(data, 1, 2)
    # (# of data, leads, points)

    if leads_seperate == True:
        if data.shape[1] > 2:
            data = np.reshape(data, (data.shape[0] * data.shape[1], 1, data.shape[2]))

    if dns:
        # data denoise using upscale and downscale back
        scale = 1151.79375 / 174.08 # LUDB data average amplitude and IEC data average amplitude
        data *= scale
        dnsigs = []
        for i in range(data.shape[0]):
            dnsigs.append(ekg_denoise(data[i]))
        dnsigs = np.array(dnsigs)[:, :, :4992]
        dnsigs = torch.Tensor(dnsigs)
        dnsigs /= scale
        return dnsigs

    if smooth:
        smoothed = []
        for i in range(data.shape[0]):
            smoothed.append(smooth_signal(data[i, 0, :], window_len=10))

        smoothed = np.array(smoothed)[:, np.newaxis, :4992]
        smoothed = torch.Tensor(smoothed)

        return smoothed

    else:
        #data = data[:, :, 500:4500]
        data = data[:, :, :4992]
        
        data = torch.Tensor(data)

        return data

def signal_augmentation(sigs, gaussian_noise_sigma=0.1):
    """
    data augmentation by adding gaussian noise.

    Args:
        sigs: (numpy) with sized [batch_size, #channels, signal length]
    Returns:
        sigs: (numpy) with sized [batch_size, #channels, signal_length]
    """
    if gaussian_noise_sigma != 0:
        noise = np.random.normal(0, gaussian_noise_sigma, size=sigs.shape)
        noisy_sigs = sigs.copy() + noise
        sigs = noisy_sigs
        #sigs = np.concatenate((sigs, noisy_sigs), 0)
    return sigs

def entropy_calc(data, base=None):
    """
    *experimental*
    calculate the shannon entropy of one signal

    Args:
        data: (Array) with sized [signal_length]
        base: (float) the logarithmic base to use
    Return:
        entropy

    """
    value, counts = np.unique(data, return_counts=True)
    return entropy(counts, base=base)

def signal_rdp(signal, epsilon=10):
    """
    using rdp algorithm to simplify the curve line.

    Args:
        signal: (list) with sized [signal_length]
        epsilon: (float) distance threshold for rdp
    Return:
        ret: (list) with sized [#simplify_points], return the simplified points' index.
    """
    process_data = []
    for i in range(len(signal)):
        process_data.append([i, signal[i]])
    processed = rdp(process_data, epsilon=epsilon)
    ret = []
    for i in range(len(processed)):
        ret.append(int(processed[i][0]))
    return ret