import torch
import torch.nn as nn
from model.RetinaNet import RetinaNet
from data.BBoxDataset import BBoxDataset
from loss.FocalLoss import FocalLoss
from data.encoder import DataEncoder
from utils.data_utils import onset_offset_generator, box_to_sig_generator, one_hot_embedding
from utils.val_utils import validation_accuracy
import viz
import os
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
torch.cuda.empty_cache()

wandb_config = {
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 5000
}

def train_model(val_ratio=0.2, test_ratio=0.2):
    model = RetinaNet(3)
    model = nn.DataParallel(model).cuda()
    model.train()

    wandb.watch(model)

    ds = BBoxDataset()
    test_len = int(len(ds) * test_ratio)
    val_len = int(len(ds) * val_ratio)
    train_len = len(ds) - test_len - val_len
    trainingset, valset, testset = torch.utils.data.random_split(ds, [train_len, val_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    valloader = torch.utils.data.DataLoader(valset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=config['lr'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=wandb.config.lr)
    criterion = FocalLoss()

    best_F1 = 0

    for epoch in range(wandb.config.epochs):
        model.train()
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets, boxes, labels) in enumerate(trainloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            loc_targets = torch.autograd.Variable(loc_targets.cuda())
            cls_targets = torch.autograd.Variable(cls_targets.cuda())
            # channel = 1
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()

            loc_preds, cls_preds = model(inputs)
            loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            loss.backward()
            optimizer.step()
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
        
        print("epoch: {}, loc_loss: {}, cls_loss: {}, total_loss: {}".format(epoch, total_loc_loss, total_cls_loss, total_loss))
        wandb.log({"epoch": epoch, "loc_loss": total_loc_loss, "cls_loss": total_cls_loss, "total_loss": total_loss})

        if epoch % 1 == 0:
            Se, PPV, F1 = eval_model(model, valloader)
            if F1 > best_F1:
                best_F1 = F1
                torch.save(model.module.state_dict(), "weights/retinanet_best.pkl")


def eval_model(model, dataloader):
    input_length = 3968
    model.eval()
    
    pred_sigs = []
    gt_sigs = []
    sigs = []
    for batch_idx, (inputs, loc_targets, cls_targets, gt_boxes, gt_labels) in enumerate(dataloader):
        batch_size = inputs.size(0)
        inputs = torch.autograd.Variable(inputs.cuda())
        loc_targets = torch.autograd.Variable(loc_targets.cuda())
        cls_targets = torch.autograd.Variable(cls_targets.cuda())
        inputs = inputs.unsqueeze(1)
        sigs.append(inputs)

        loc_preds, cls_preds = model(inputs)

        loc_preds = loc_preds.data.squeeze().type(torch.FloatTensor) # sized [#anchors * 3, 2]
        cls_preds = cls_preds.data.squeeze().type(torch.FloatTensor) # sized [#ahchors * 3, 3]

        loc_targets = loc_targets.data.squeeze().type(torch.FloatTensor)
        cls_targets = cls_targets.data.squeeze().type(torch.LongTensor)

        #if batch_idx == 0:
            #print(cls_targets)
        # decoder only process data 1 by 1.
        encoder = DataEncoder()
        for i in range(batch_size):
            boxes, labels, sco, is_found = encoder.decode(loc_preds[i], cls_preds[i], input_length)

        #ground truth decode using another method
            gt_boxes_tensor = torch.tensor(gt_boxes[i])
            gt_labels_tensor = torch.tensor(gt_labels[i])
            xmin = gt_boxes_tensor[:, 0].clamp(min=1)
            xmax = gt_boxes_tensor[:, 1].clamp(max=input_length - 1)
            gt_sig = box_to_sig_generator(xmin, xmax, gt_labels_tensor, input_length, background=False)
            
            if is_found:
                boxes = boxes.ceil()
                xmin = boxes[:, 0].clamp(min = 1)
                xmax = boxes[:, 1].clamp(max = input_length - 1)

                # there is no background anchor on predict labels
                pred_sig = box_to_sig_generator(xmin, xmax, labels, input_length, background=False)
            else:
                pred_sig = torch.zeros(1, 4, input_length)

            pred_sigs.append(pred_sig)
            gt_sigs.append(gt_sig)
    sigs = torch.cat(sigs, 0)
    pred_signals = torch.cat(pred_sigs, 0)
    gt_signals = torch.cat(gt_sigs, 0)
    plot = viz.predict_plotter(sigs[0][0], pred_signals[0], gt_signals[0])
    wandb.log({"visualization": plot})
    pred_onset_offset = onset_offset_generator(pred_signals)
    gt_onset_offset = onset_offset_generator(gt_signals)
    TP, FP, FN = validation_accuracy(pred_onset_offset, gt_onset_offset)

    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * Se * PPV / (Se + PPV)

    print("Se: {} PPV: {} F1 score: {}".format(Se, PPV, F1))
    wandb.log({"Se": Se, "PPV": PPV, "F1": F1})
    
    return Se, PPV, F1

ex = wandb.init(project="PQRST-segmentation")
ex.config.setdefaults(wandb_config)

train_model()