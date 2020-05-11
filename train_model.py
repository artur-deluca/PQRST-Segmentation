import torch
from model.RetinaNet import RetinaNet
from data.BBoxDataset import BBoxDataset
from loss.FocalLoss import FocalLoss
from data.encoder import DataEncoder
from utils.data_utils import onset_offset_generator, box_to_sig_generator, one_hot_embedding
from utils.val_utils import validation_accuracy

config = {
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 5000
}


def train_model(val_ratio=0.2, test_ratio=0.2):
    model = RetinaNet(3).cuda()
    model.train()

    ds = BBoxDataset()
    test_len = int(len(ds) * test_ratio)
    val_len = int(len(ds) * val_ratio)
    train_len = len(ds) - test_len - val_len
    trainingset, valset, testset = torch.utils.data.random_split(ds, [train_len, val_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=config['batch_size'], shuffle=True, collate_fn=ds.collate_fn)
    train_as_test_loader = torch.utils.data.DataLoader(trainingset, batch_size=1, shuffle=True, collate_fn=ds.collate_fn)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, collate_fn=ds.collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True, collate_fn=ds.collate_fn)
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['lr'])
    criterion = FocalLoss()

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
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
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print("batch_idx: {}, loc_loss: {}, cls_loss: {}, train_loss: {}, avg_loss: {}".format(batch_idx, loc_loss.item(), cls_loss.item(), loss.item(),total_loss/(batch_idx+1)))
        eval_model(model, train_as_test_loader)
        eval_model(model, valloader)


def eval_model(model, dataloader):
    input_length = 4992
    model.eval()
    
    pred_sigs = []
    gt_sigs = []
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(dataloader):
        inputs = torch.autograd.Variable(inputs.cuda())
        loc_targets = torch.autograd.Variable(loc_targets.cuda())
        cls_targets = torch.autograd.Variable(cls_targets.cuda())
        inputs = inputs.unsqueeze(1)

        loc_preds, cls_preds = model(inputs)

        loc_preds = loc_preds.data.squeeze().type(torch.FloatTensor) # sized [#anchors * 3, 2]
        cls_preds = cls_preds.data.squeeze().type(torch.FloatTensor) # sized [#ahchors * 3, 3]

        loc_targets = loc_targets.data.squeeze().type(torch.FloatTensor)
        cls_targets = cls_targets.data.squeeze().type(torch.LongTensor)

        # decoder only process data 1 by 1.
        encoder = DataEncoder()
        boxes, labels, sco, is_found = encoder.decode(loc_preds, cls_preds, input_length)
        gt_boxes, gt_labels, gt_sco, gt_is_found = encoder.decode(loc_targets, one_hot_embedding(cls_targets, 4), input_length)

        if is_found:
            boxes = boxes.ceil()
            xmin = boxes[:, 0].clamp(min = 1)
            xmax = boxes[:, 1].clamp(max = input_length - 1)
        
            pred_sig = box_to_sig_generator(xmin, xmax, labels, input_length)
        else:
            pred_sig = torch.zeros(1, 4, input_length)
        gt_sig = box_to_sig_generator(gt_boxes[:, 0], gt_boxes[:, 1], gt_labels, input_length)
        pred_sigs.append(pred_sig)
        gt_sigs.append(gt_sig)
    pred_onset_offset = onset_offset_generator(torch.cat(pred_sigs, 0))
    gt_onset_offset = onset_offset_generator(torch.cat(gt_sigs, 0))

    TP, FP, FN = validation_accuracy(pred_onset_offset, gt_onset_offset)

    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 * Se * PPV / (Se + PPV)

    print("Se: {} PPV: {} F1 score: {}".format(Se, PPV, F1))


train_model()