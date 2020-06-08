import torch
import torch.nn as nn
import os
import wandb

from model.RetinaNet import RetinaNet
from data.BBoxDataset import BBoxDataset
from loss.FocalLoss import FocalLoss
from utils.val_utils import eval_retinanet, eval_retinanet_elementwise
from test.test_retinanet import test_retinanet_using_IEC

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

wandb_config = {
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 1500,
    "cls_scale": 1,
    "loc_scale": 1,
    "train_denoise": False,
    "test_denoise": False,
    "augmentation_gaussian_noise_sigma": 0.1,

    "n_model_ensemble": 1,
}

def train_model(val_ratio=0.2, test_ratio=0.2):
    ensemble_model_list = [nn.DataParallel(RetinaNet(3)).cuda() for x in range(wandb.config.n_model_ensemble)]
    #model = RetinaNet(3)
    #model = nn.DataParallel(model).cuda()
    #model.train()

    ds = BBoxDataset(wandb.config.train_denoise)
    test_len = int(len(ds) * test_ratio)
    val_len = int(len(ds) * val_ratio)
    train_len = len(ds) - test_len - val_len
    trainingset, valset, testset = torch.utils.data.random_split(ds, [train_len, val_len, test_len])
    valloader = torch.utils.data.DataLoader(valset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    
    for i in range(wandb.config.n_model_ensemble):
        trainloader = torch.utils.data.DataLoader(trainingset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    
        optimizer = torch.optim.Adam(params=ensemble_model_list[i].parameters(), lr=wandb.config.lr)
        criterion = FocalLoss()

        best_F1 = 0

        for epoch in range(wandb.config.epochs):
            ensemble_model_list[i].train()
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

                loc_preds, cls_preds = ensemble_model_list[i](inputs)
                loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                loss = wandb.config.loc_scale * loc_loss + wandb.config.cls_scale * cls_loss
                loss.backward()
                optimizer.step()
                total_loc_loss += loc_loss.item()
                total_cls_loss += cls_loss.item()
                total_loss += loss.item()
            
            print("epoch: {}, loc_loss: {}, cls_loss: {}, total_loss: {}".format(epoch, total_loc_loss, total_cls_loss, total_loss))
            wandb.log({"epoch": epoch, "loc_loss": total_loc_loss, "cls_loss": total_cls_loss, "total_loss": total_loss})

            if epoch % 1 == 0 and epoch >= 0:
                Se, PPV, F1 = eval_retinanet(model, valloader)
                if F1 > best_F1:
                    best_F1 = F1
                    torch.save(model.module.state_dict(), "weights/retinanet_best.pkl")
                test_IEC_result = test_retinanet_using_IEC(model)
                print("IEC: {}".format(test_IEC_result))

            if epoch % 100 == 0 and epoch >= 0:
                print("------Start checking overfitting...------")
                Se, PPV, F1 = eval_retinanet(model, trainloader)
                
        result = eval_retinanet_elementwise(ensemble_model_list[i], trainloader)

            



def train():
    ex = wandb.init(project="PQRST-segmentation")
    ex.config.setdefaults(wandb_config)

    train_model()