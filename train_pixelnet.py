import torch
import torch.nn as nn
from model.PixelNet import PixelNet
import data_generator as g
from utils.data_utils import onset_offset_generator, box_to_sig_generator, one_hot_embedding
from utils.val_utils import validation_accuracy
from torch.utils.data import DataLoader, random_split
from eval import eval_net

config = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 5000
}


def train_model(val_ratio=0.2, test_ratio=0.2, device=torch.device('cuda')):
    batch_size = config['batch_size']
    data = g.load_dataset_with_origin_length()
    # calculate train, validation, test dataset size
    n_test = int(len(data) * test_ratio)
    n_val = int(len(data) * val_ratio)
    n_train = len(data) - n_test - n_val
    train, val, test = random_split(data, [n_train, n_val, n_test])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = PixelNet(3, 4992).cuda()
    model.train()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = torch.autograd.Variable(x.cuda())
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch: {}, total_loss: {}".format(epoch, total_loss))
        val_score, acc, Se, PPV, F1, interval = eval_net(model,val_loader, device=device)
        print("Validation Acc:\t{}".format(acc))
        print("Se:\t{}".format(Se))
        print("PPV:\t{}".format(PPV))
        print("F1:\t{}".format(F1))

if __name__ == "__main__":
    train_model()