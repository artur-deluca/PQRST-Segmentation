import sys
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from torch import optim
from tqdm import tqdm

from model import UNet
from utils.val_utils import eval_unet
from utils.data_utils import load_dataset_using_pointwise_labels
from test import test_unet

wandb_config = {
        "epochs": 200,
        "batch_size": 32,
        "lr": 1e-4,
        }


def train_model(net, epochs=6000, batch_size=32, lr=1e-4, device=torch.device('cuda')):
    """
    training the UNet model

    Args:
        net: (nn.Module) UNet module
        epochs: (int) training epochs
        batch_size: (int) batch_size
        lr: (float) learning rate
        device: (torch.device) execute device. cuda/cpu
    """
    wandb.watch(net)

    data = load_dataset_using_pointwise_labels()

    # calculate train, validation, test dataset size
    n_test = int(len(data) * 0.3)
    n_val = int(len(data) * 0.1)
    n_train = len(data) - n_test - n_val
    train, val, test = random_split(data, [n_train, n_val, n_test])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    global_step = 0

    logging.info(f'''Start training:
        Epochs:         {epochs}
        Batch size:     {batch_size}
        Learning rate:  {lr}
        Training size:  {n_train}
        Validation size:{n_val}
        Device:         {device.type}
        ''')

    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(alpha=wandb.config.focalloss_alpha,gamma=wandb.config.focalloss_gamma)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        with tqdm(total=n_train, ncols=100) as pbar:
            for batch in train_loader:
                x = batch[0]
                ground_truth = batch[1]

                x = x.to(device, dtype=torch.float32)
                ground_truth = ground_truth.to(device, dtype=torch.float32)

                pred = net(x)
                loss = criterion(pred, ground_truth)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_description('Epoch %d / %d' % (epoch + 1, epochs))
                pbar.update(x.shape[0])
                tqdm._instances.clear()
                global_step += 1

                #if global_step % (len(data) // (10 * batch_size)) == 0:
            val_score, acc, Se, PPV, F1, interval = eval_unet(net, val_loader, device)
            scheduler.step(val_score)
            wandb.log({'epoch': epoch, 'loss': val_score, 'Se': Se, 'PPV': PPV, 'F1': F1})
            #wandb.log(interval)
            logging.info('Validation cross entropy: {}'.format(val_score))
            logging.info('Acc:\t{}'.format(acc))
            logging.info('Se:\t{}'.format(Se))
            logging.info('PPV:\t{}'.format(PPV))
            logging.info('F1:\t{}'.format(F1))

            #if epoch % 10 == 0:
                #test_net(net, test_loader, device)

    test_iter = iter(test_loader)
    x, y = test_iter.next()
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)
    plot, interval = test_unet.test(net, x, y)
    wandb.log({'visualization': plot})

    torch.save(net.state_dict(), "model.pkl")

    torch.save(net.state_dict(), os.path.join(wandb.run.dir, "model.pkl"))


def train():
    ex = wandb.init(project="PQRST-segmentation")
    ex.config.setdefaults(wandb_config)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(in_ch=1, out_ch=4)
    net.to(device)

    try:
        train_model(net=net, device=device, batch_size=wandb.config.batch_size, lr=wandb.config.lr, epochs=wandb.config.epochs)
    except KeyboardInterrupt:
        try:
            save = input("save?(y/n)")
            if save == "y":
                torch.save(net.state_dict(), 'net_params.pkl')
            sys.exit(0)
        except SystemExit:
            os._exit(0)
