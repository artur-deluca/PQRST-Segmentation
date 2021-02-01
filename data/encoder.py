import math
import torch

from utils.data_utils import box_iou, box_nms, change_box_order

class DataEncoder:
    def __init__(self):
        self.anchor_areas = [8, 16, 32, 64]
        # Note that there is no aspect ratio on 1D data
        self.scale_ratios = [0.75, 1.0, 1.5]
        # Every feature maps' anchors will have 3 different size of bbox
        self.anchor_length = self._get_anchor_length()

    def _get_anchor_length(self):
        """
        Compute anchor length for each feature map

        Returns:
            anchor_length: (tensor) anchor length, sized [#fm, #anchors_per_cell, 1]
        """
        anchor_length = []
        for s in self.anchor_areas:
            for sr in self.scale_ratios:
                anchor_l = s * sr
                anchor_length.append(anchor_l)
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_length).view(num_fms, -1, 1)


    def _get_anchor_boxes(self, input_size):
        """
        compute anchor boxes for each feature map
        
        Args:
            input_size: (tensor) model input size of (length)
        
        Returns:
            boxes: (list) anchor boxes for each feature map. Each of size (#anchors, 2),
                        where #ahchors = fml * #anchors_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2., i+3)).ceil() for i in range(num_fms)] # p3 ~ p6 feature map sizes
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_l = int(fm_size)
            #print(fm_l)
            #print(grid_size)
            x = torch.arange(0, fm_l) + 0.5 # ancher position [fm_l]
            x = (x * grid_size).view(fm_l, 1, 1).expand(fm_l, 3, 1)
            length = self.anchor_length[i].view(1, 3, 1).expand(fm_l, 3, 1)
            #print(self.anchor_length[i])
            box = torch.cat([x, length], 2) # [x, length]
            boxes.append(box.view(-1, 2))
        return torch.cat(boxes, 0)
    
    def encode(self, boxes, labels, input_size):
        """
        Encode target bounding boxes and class labels.

        tx = (x - anchor_x) / anchor_l
        tl = log(l / anchor_l.)

        Args:
            boxes: (tensor) bounding boxes of (x_min, x_max), sized [#obj, 2]
            labels: (tensor) object class labels, sized [#obj, ]
            input_size: (int/tuple) model input size of [x]

        Returns:
            loc_targets: (tensor) encoded bounding boxes, sized [#anchors, 2]
            cls_targets: (tensor) encoded class labels, sized [#anchors, ]
        """
        input_size = torch.Tensor([input_size])
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xx2xl')
        ious = box_iou(anchor_boxes, boxes, order='xl')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]
        loc_x = (boxes[:, :1] - anchor_boxes[:, :1]) / anchor_boxes[:, 1:]
        loc_l = torch.log(boxes[:, 1:]/anchor_boxes[:, 1:])
        loc_targets = torch.cat([loc_x, loc_l], 1)
        cls_targets = 1 + labels[max_ids]


        # background targets
        cls_targets[max_ious <= 0.4] = 0
        ignore = (max_ious > 0.4) & (max_ious < 0.5)
        cls_targets[ignore] = -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size, CLS_THRESH=0.5, NMS_THRESH=0.5):
        """
        Decode outputs back to bounding box locations and class labels

        Args:
            loc_preds: (tensor) predicted locations, sized [#anchors, 2]
            cls_preds: (tensor) predicted class labels, sized [#anchors, #classes]
            input_size: (input/tuple) model input size of (l).
            CLS_THRESH: (float) determine how high the box score to choose.
            NMS_THRESH: (float) determine how large ious to suppress
        
        Returns:
            boxes: (tensor) decode box locations, sized [#obj, 2]
            labels: (tensor) class labels for each box, sized [#obj, ]
        
        (background label is 0)
        """
        input_size = torch.Tensor([input_size])
        anchor_boxes = self._get_anchor_boxes(input_size)
        loc_x = loc_preds[:, :1]
        loc_l = loc_preds[:, 1:]

        x = loc_x * anchor_boxes[:, 1:] + anchor_boxes[:, :1]
        l = loc_l.exp() * anchor_boxes[:, 1:]
        boxes = torch.cat([x-l/2, x+l/2], 1)
        score, labels = cls_preds.sigmoid().max(1)
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()
        if not ids.size():
            return torch.tensor([0,0]), torch.tensor([0]), torch.tensor([0]), False
        if ids.size(0) == 0:
            return torch.tensor([0,0]), torch.tensor([0]), torch.tensor([0]), False

        keep, sco = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep], sco, True