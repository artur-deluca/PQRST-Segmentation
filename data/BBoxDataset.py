import torch
import numpy as np
from data.encoder import DataEncoder
from utils.data_utils import load_raw_dataset_and_bbox_labels, load_raw_dataset_and_bbox_labels_CAL, ekg_denoise, signal_augmentation
from utils.data_utils import normalize as Normalize
from os import path
import wandb

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

class BBoxDataset(torch.utils.data.Dataset):
    """
    define the labels that are used in object detection model like RetinaNet
    """
    def __init__(self, denoise=True):
        self.signals = []
        self.boxes = []
        self.labels = []
        self.peaks = []

        self.num_samples = 0
        
        self.raw_dataset_path = config["General"]["LUDB_path"]

        self.encoder = DataEncoder()
        self.get_signal_annotations(leads_seperate=True, normalize=True, denoise=denoise, gaussian_noise_sigma=wandb.config.augmentation_gaussian_noise_sigma, data_augmentation=wandb.config.data_augmentation)
        

    def get_signal_annotations(self, leads_seperate=True, normalize=True, denoise=True, gaussian_noise_sigma=0.1, data_augmentation=True):
        """
        compute and save the bbox result in dataset
        
        Args:
            leads_seperate: (bool) seperate leads or not
            normalize: (bool) normalize the signal or not
            denoise: (bool) denoise the signal using wavelet thresholding or not
            gaussian_noise_sigma: (float) the noise sigma add to data augmentation, if equals 0, then there will be no data augmentation
            data_augmentation: (bool) use data augmentation that scale the signal on different segments or not. 
        
        signals:    (Tensor) with sized [#signal, signal_length]
        boxes:      (list) with sized [#signal, #objs, 2]
        labels:     (list) with sized [#signal, #objs, ]
        peaks:      (list) with sized [#signal, #objs, ]
        """
        if denoise and path.exists(config["RetinaNet"]["output_path"]+"LUDB_preprocessed_data_denoise.pt"):
            self.signals, self.boxes, self.labels, self.peaks, self.num_samples = torch.load(config["RetinaNet"]["output_path"]+"LUDB_preprocessed_data_denoise.pt")
        elif not denoise and path.exists(config["RetinaNet"]["output_path"]+"LUDB_preprocessed_data.pt"):
            self.signals, self.boxes, self.labels, self.peaks, self.num_samples = torch.load(config["RetinaNet"]["output_path"]+"LUDB_preprocessed_data.pt")
        else:
            signals, bboxes, labels, peaks = load_raw_dataset_and_bbox_labels(self.raw_dataset_path)
            signals_, bboxes_, labels_, peaks_ = load_raw_dataset_and_bbox_labels_CAL()

            signals.extend(signals_)
            bboxes.extend(bboxes_)
            labels.extend(labels_)
            peaks.extend(peaks_)

            # with sized [#subjects, #leads, signal_length] [#subjects, #leads, #objs, 2] [#subjects, #leads, #objs]
            if gaussian_noise_sigma != 0.0:
                signals = signal_augmentation(signals)
                bboxes_aug = bboxes.copy()
                labels_aug = labels.copy()
                peaks_aug = peaks.copy()
                bboxes = [*bboxes, *bboxes_aug]
                labels = [*labels, *labels_aug]
                peaks = [*peaks, *peaks_aug]
                
            if leads_seperate == True:
                num_subjects = len(signals)
                for i in range(num_subjects):
                    num_leads = len(signals[i])
                    if denoise:
                        d = ekg_denoise(signals[i])
                    for j in range(num_leads):
                        self.signals.append(torch.Tensor(signals[i][j])) if not denoise else self.signals.append(torch.Tensor(d[j]))
                        self.boxes.append(torch.Tensor(bboxes[i][j]))
                        self.labels.append(torch.Tensor(labels[i][j]))
                        self.peaks.append(torch.Tensor(peaks[i][j]))

                        self.num_samples += 1
            if normalize:
                self.signals = Normalize(torch.stack(self.signals), instance=True)
                
            if denoise:
                torch.save((self.signals, self.boxes, self.labels, self.peaks, self.num_samples), "./data/LUDB_preprocessed_data_denoise.pt")
            else:
                torch.save((self.signals, self.boxes, self.labels, self.peaks, self.num_samples), "./data/LUDB_preprocessed_data.pt")
        
        if data_augmentation:
            for i in range(self.signals):
                x = self.signal[i].copy()
                for j in range(len(self.boxes[i])):
                    if self.labels[i][j] == 0: # p duration
                        for k in range(self.boxes[i][j][0], self.boxes[i][j][1]):
                            x[k] *= 0.8
                self.signal.append(x)
                self.boxes.append(self.boxes[i].copy())
                self.labels.append(self.labels[i].copy())
                self.peaks.append(self.peaks[i].copy())
                self.num_samples += 1

    def __getitem__(self,idx):
        """
        Load signal

        Args:
            idx: (int) signal index
        
        Returns:
            sig:         (Tensor) signal tensor
            loc_targets: (Tensor) location targets
            cls_targets: (Tensor) class label targets
        """
        sig = self.signals[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        peaks = self.peaks[idx]

        return sig, boxes, labels, peaks

    def collate_fn(self, batch):
        """
        Encode targets
        
        Args:
            batch: (list) of signals, loc_targets, cls_targets
        """
        sigs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        peaks = [x[3] for x in batch]

        input_size = 3968 # data length

        num_sigs = len(sigs)

        inputs = torch.zeros(num_sigs, input_size)

        loc_targets = []
        cls_targets = []
        for i in range(num_sigs):
            inputs[i] = sigs[i]
            
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size)

            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), boxes, labels, peaks

    def __len__(self):
        return self.num_samples