import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import onset_offset_generator, validation_accuracy

def eval_net(net, loader, device):
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
            TP += tp
            FP += fp
            FN += fn
            
            pbar.update()
        
    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * Se * PPV / (Se + PPV)
        
    return tot / n_val, correct / total, Se, PPV, F1
