import torch
import torch.nn as nn
import numpy as np
import os
import wandb

from model.RetinaNet import RetinaNet
from data.BBoxDataset import BBoxDataset
from loss.FocalLoss import FocalLoss
from utils.val_utils import eval_retinanet
from test.test_retinanet import test_retinanet_using_IEC
from optimizer.radam import RAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

wandb_config = {
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 1500,
    "cls_scale": 1,
    "loc_scale": 1,
    "train_denoise": False,
    "test_denoise": True,
    "augmentation_gaussian_noise_sigma": 0.0,
    "data_augmentation": False,
}

def train_model(val_ratio=0.2, test_ratio=0.0):
    model = RetinaNet(3)
    model = nn.DataParallel(model).cuda()
    model.train()

    #wandb.watch(model)

    ds = BBoxDataset(wandb.config.train_denoise)
    test_len = int(len(ds) * test_ratio)
    val_len = int(len(ds) * val_ratio)
    train_len = len(ds) - test_len - val_len
    trainingset, valset, testset = torch.utils.data.random_split(ds, [train_len, val_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    valloader = torch.utils.data.DataLoader(valset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=config['lr'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=wandb.config.lr)
    #optimizer = RAdam(params=model.parameters(), lr=wandb.config.lr)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    criterion = FocalLoss()

    best_F1 = 0
    best_IEC = 0

    for epoch in range(wandb.config.epochs):
        model.train()
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets, boxes, labels, peaks) in enumerate(trainloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            loc_targets = torch.autograd.Variable(loc_targets.cuda())
            cls_targets = torch.autograd.Variable(cls_targets.cuda())
            # channel = 1
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()

            loc_preds, cls_preds = model(inputs)
            loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = wandb.config.loc_scale * loc_loss + wandb.config.cls_scale * cls_loss
            loss.backward()
            optimizer.step()
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
        #lr_scheduler.step()
        
        print("epoch: {}, loc_loss: {}, cls_loss: {}, total_loss: {}".format(epoch, total_loc_loss, total_cls_loss, total_loss))
        wandb.log({"epoch": epoch, "loc_loss": total_loc_loss, "cls_loss": total_cls_loss, "total_loss": total_loss})

        if epoch % 1 == 0 and epoch >= 0:
            Se, PPV, F1 = eval_retinanet(model, valloader)
            #lr_scheduler.step(F1)
            if F1 > best_F1:
                best_F1 = F1
                torch.save(model.module.state_dict(), "weights/retinanet_best.pkl")
                torch.save(model.module.state_dict(), os.path.join(wandb.run.dir, "model_best.pkl"))
            test_IEC_result, pass_result = test_retinanet_using_IEC(model)
            if np.mean(test_IEC_result) >= best_IEC:
                best_IEC = np.mean(test_IEC_result)
                torch.save(model.module.state_dict(), "weights/retinanet_best_IEC.pkl")
                torch.save(model.module.state_dict(), os.path.join(wandb.run.dir, "model_best_IEC.pkl"))
            if np.all(pass_result==['Passed','Passed','Passed','Passed']):
                torch.save(model.module.state_dict(), "weights/retinanet_pass_all_IEC.pkl")
            print("IEC: {}".format(test_IEC_result))

        if epoch % 100 == 0 and epoch >= 0:
            print("------Start checking overfitting...------")
            Se, PPV, F1 = eval_retinanet(model, trainloader)


def train():
    ex = wandb.init(project="PQRST-segmentation")
    ex.config.setdefaults(wandb_config)

    train_model()