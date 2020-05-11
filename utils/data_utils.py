import numpy as np
import torch
import json

leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

def load_raw_dataset(raw_dataset):
    """
    Get raw dataset with signals, bboxes and labels
    
    Args:
        raw_dataset: (string) dataset path
    
    Returns:
        X: (list) input signals with sized [#subjects, #leads, signal_length=5000]
        BBoxes: (list) target bboxes with sized [#subjects, #leads, #objs, 2]
        Labels: (list) target labels with sized [#subjects, #leads, #objs] 
    """
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    BBoxes = []
    Labels= []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        bbox = []
        label = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'][:4992])

            delineation_tables = leads[leads_names[0]]['DelineationDoc']
            
            p_delin = delineation_tables['p']
            qrs_delin = delineation_tables['qrs']
            t_delin = delineation_tables['t']

            # background label 0 will add when encoding
            p_boxes, p_labels = get_bbox_labels(p_delin, 0)
            qrs_boxes, qrs_labels = get_bbox_labels(qrs_delin, 1)
            t_boxes, t_labels = get_bbox_labels(t_delin, 2)
            
            b = [*p_boxes, *qrs_boxes, *t_boxes]
            l = [*p_labels, *qrs_labels, *t_labels]
            
            bbox.append(b)
            label.append(l)

        X.append(x)
        BBoxes.append(bbox)
        Labels.append(label)

    return X, BBoxes, Labels

def get_bbox_labels(delineation, label):
    """
    Args: 
        delineation: (list) sized [#obj, 3], 3 indicates onset, peak, offset
    Returns:
        bboxes: (list) with sized [#obj, 2], 2 indicates (xmin, xmax)
        labels: (list) with sized [#obj]
    """
    bboxes = []
    labels = []
    for obj in delineation:
        xmin = obj[0]
        xmax = obj[2]
        bboxes.append((xmin, xmax))
        labels.append(label)
    return bboxes, labels

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

def box_to_sig_generator(xmin, xmax, labels, sig_length):
    """
    Args:
        xmin: (tensor) with sized [#obj]
        xmax: (tensor) with sized [#obj]
        labels: (tensor) with sized [#obj]
        sig_length: (int) signal length
    Returns:
        sig: (tensor) with sized [1, 4, sig_length]
    """
    xmin = xmin.long()
    xmax = xmax.long()
    labels = labels.long()
    sig = torch.zeros(1, 4, sig_length)
    for i in range(len(labels)):
        sig[0, labels[i], xmin[i]:xmax[i]] = 1
    return sig

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

def change_box_order(boxes, order):
    """change box order between (x_min, x_max) and (x_center, length)

    Args:
        boxes: (tensor) bounding boxes, sized [N, 2]
        order: (str) either 'xx2xl' or 'xl2xx'
    
    Returns:
        (tensor) converted bounding boxes, sized [N, 2]
    """
    assert order in ['xx2xl', 'xl2xx']
    a = boxes[:, :1]
    b = boxes[:, 1:]
    if order == 'xx2xl':
        return torch.cat([(a+b)/2, b-a+1], 1)
    return torch.cat([a-b/2, a+b/2], 1)

def box_iou(box1, box2, order='xx'):
    """Compute the intersection over union of two set of boxes

    The default box order is (xmin, xmax)

    Args:
        box1: (tensor) bounding boxes, sized [N, 2]
        box2: (tensor) bounding boxes, sized [M, 2]
        order: (str) box order, either 'xx' or 'xl'
    
    Returns:
        (tensor) iou, sized[N, M]
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
        bboxes: (tensor) bounding boxes, sized [N, 4]
        scores: (tensor) bbox scores, szied [N, ]
        threshold: (float) overlap threshold
        mode: (str) 'union' or 'min'

    Returns:
        keep: (tensor) selected indices
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
    :param labels: (LongTensor) class label, sized [N,].
    :param num_classes: (int) number of classes.
    :return:
            (tensor) encoded labels, size [N, #classes].
    """
    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]